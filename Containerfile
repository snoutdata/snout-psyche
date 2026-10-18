# snout-psyche's image: one static binary on scratch.
#
#   docker buildx build -f Containerfile --platform linux/arm64 -t snout-psyche .
#
# The context is the repository root, for its lockfile (in the SnoutData monorepo, the stack
# workspace root, with `-f psyche/Containerfile`). Scratch, because nothing runs in this
# container but the server: TLS roots are compiled in (webpki-roots), names resolve through the
# resolv.conf the runtime mounts, and there is no shell for anything to be run with.
FROM docker.io/library/rust:1.98.1-alpine AS build
RUN apk add --no-cache musl-dev
WORKDIR /src
COPY . .
RUN cargo build --locked --release -p snout-psyche \
	&& cp target/release/snout-psyche /snout-psyche

FROM scratch
COPY --from=build /snout-psyche /snout-psyche
USER 1000:1000
EXPOSE 5300
# How SnoutData Studio's "Find databases" knows this container is part of the SnoutData stack:
# by label, never by guessing from the image name. Only the `postgres` component is offered as a
# database; the rest are recognised and left out.
LABEL com.snoutdata.stack="1" com.snoutdata.component="psyche"
ENTRYPOINT ["/snout-psyche"]
