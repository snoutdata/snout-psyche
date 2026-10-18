//! Who a call is from, and whom a budget or a limit is for.
//!
//! A call is from the project's service key, a key, or a signed-in user of the project's app, and a
//! key may belong to a user (its `owner`). A user is in at most one group (`psyche.members`). A
//! budget or a limit is for the project, one key, every user (each on their own), one user, or a
//! group; one naming a user replaces the every-user ones for that user, so a user can be given more
//! (or less) than the default.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use crate::keys::KeyId;

/// A user of the project's app: `auth.users.id`, the token's `sub`.
pub type UserId = KeyId;
/// A group of users, `psyche.groups.id`.
pub type GroupId = KeyId;

/// Who a call is from, as budgets, limits and the request log see it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Who {
	pub key: Option<KeyId>,
	pub user: Option<UserId>,
	pub group: Option<GroupId>,
}

/// Whom a budget or a limit is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Scope {
	Project,
	Key(KeyId),
	/// Every user, each on their own.
	EveryUser,
	User(UserId),
	Group(GroupId),
}

/// What spend and buckets are kept per: a scope resolved for one call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Subject {
	Project,
	Key(KeyId),
	User(UserId),
	Group(GroupId),
}

impl Scope {
	/// What this scope holds a call by `who` to, if it holds it at all. `overridden` is the users
	/// with a scope of their own, whom the every-user scope leaves alone.
	pub fn subject(self, who: &Who, overridden: &HashSet<UserId>) -> Option<Subject> {
		match self {
			Scope::Project => Some(Subject::Project),
			Scope::Key(key) => (who.key == Some(key)).then_some(Subject::Key(key)),
			Scope::EveryUser => who
				.user
				.filter(|user| !overridden.contains(user))
				.map(Subject::User),
			Scope::User(user) => (who.user == Some(user)).then_some(Subject::User(user)),
			Scope::Group(group) => (who.group == Some(group)).then_some(Subject::Group(group)),
		}
	}

	/// From a row's `key_id`, `every_user`, `user_id` and `group_id`, at most one of which is set.
	pub fn from_row(
		key: Option<&str>,
		every_user: bool,
		user: Option<&str>,
		group: Option<&str>,
	) -> Option<Scope> {
		Some(match (key, every_user, user, group) {
			(None, false, None, None) => Scope::Project,
			(Some(key), false, None, None) => Scope::Key(KeyId::parse(key)?),
			(None, true, None, None) => Scope::EveryUser,
			(None, false, Some(user), None) => Scope::User(KeyId::parse(user)?),
			(None, false, None, Some(group)) => Scope::Group(KeyId::parse(group)?),
			_ => return None,
		})
	}
}

/// The subjects a call by `who` is counted under, whether or not a budget names them: when any
/// budget holds the call, its spend is kept for each, so a budget added later for one of them sees
/// what was spent before it (and a restart reads all of it from the daily totals).
pub fn subjects(who: &Who) -> Vec<Subject> {
	let mut out = vec![Subject::Project];
	out.extend(who.key.map(Subject::Key));
	out.extend(who.user.map(Subject::User));
	out.extend(who.group.map(Subject::Group));
	out
}

/// The users some scope in `scopes` names one by one.
pub fn overridden(scopes: impl Iterator<Item = Scope>) -> HashSet<UserId> {
	scopes
		.filter_map(|scope| match scope {
			Scope::User(user) => Some(user),
			_ => None,
		})
		.collect()
}

/// Which group each user is in, from `psyche.members`, held in memory like the keys.
#[derive(Clone, Default)]
pub struct Members(Arc<RwLock<HashMap<UserId, GroupId>>>);

impl std::fmt::Debug for Members {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.write_str("Members(..)")
	}
}

impl Members {
	pub fn group_of(&self, user: UserId) -> Option<GroupId> {
		self.0
			.read()
			.unwrap_or_else(|e| e.into_inner())
			.get(&user)
			.copied()
	}

	pub fn replace(&self, members: HashMap<UserId, GroupId>) {
		*self.0.write().unwrap_or_else(|e| e.into_inner()) = members;
	}

	/// Who a call by `user` (or by a key `user` owns) is from.
	pub fn who(&self, key: Option<KeyId>, user: Option<UserId>) -> Who {
		Who {
			key,
			user,
			group: user.and_then(|user| self.group_of(user)),
		}
	}
}

/// Every membership, from `psyche.members`.
pub async fn load(
	client: &tokio_postgres::Client,
) -> Result<HashMap<UserId, GroupId>, tokio_postgres::Error> {
	let rows = client
		.query(
			"SELECT user_id::text, group_id::text FROM psyche.members",
			&[],
		)
		.await?;
	Ok(rows
		.iter()
		.filter_map(|row| Some((KeyId::parse(row.get(0))?, KeyId::parse(row.get(1))?)))
		.collect())
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_user_of_their_own_leaves_the_every_user_scope() {
		let (alice, bob, team, key) = (
			KeyId([1; 16]),
			KeyId([2; 16]),
			KeyId([3; 16]),
			KeyId([4; 16]),
		);
		let overridden = overridden([Scope::User(alice), Scope::EveryUser].into_iter());
		let by_alice = Who {
			key: None,
			user: Some(alice),
			group: Some(team),
		};
		let by_bob = Who {
			key: Some(key),
			user: Some(bob),
			group: None,
		};
		assert_eq!(Scope::EveryUser.subject(&by_alice, &overridden), None);
		assert_eq!(
			Scope::EveryUser.subject(&by_bob, &overridden),
			Some(Subject::User(bob))
		);
		assert_eq!(
			Scope::User(alice).subject(&by_alice, &overridden),
			Some(Subject::User(alice))
		);
		assert_eq!(Scope::User(alice).subject(&by_bob, &overridden), None);
		assert_eq!(
			Scope::Group(team).subject(&by_alice, &overridden),
			Some(Subject::Group(team))
		);
		assert_eq!(Scope::Group(team).subject(&by_bob, &overridden), None);
		assert_eq!(
			Scope::Key(key).subject(&by_bob, &overridden),
			Some(Subject::Key(key))
		);
		assert_eq!(
			Scope::Project.subject(&Who::default(), &overridden),
			Some(Subject::Project)
		);
		assert_eq!(
			subjects(&by_alice),
			vec![Subject::Project, Subject::User(alice), Subject::Group(team)]
		);
	}

	#[test]
	fn a_row_names_one_subject_at_most() {
		let id = "00000000-0000-0000-0000-000000000001";
		assert_eq!(
			Scope::from_row(None, false, None, None),
			Some(Scope::Project)
		);
		assert_eq!(
			Scope::from_row(None, true, None, None),
			Some(Scope::EveryUser)
		);
		assert!(matches!(
			Scope::from_row(None, false, Some(id), None),
			Some(Scope::User(_))
		));
		assert_eq!(Scope::from_row(Some(id), true, None, None), None);
		assert_eq!(Scope::from_row(None, false, Some("nope"), None), None);
	}
}
