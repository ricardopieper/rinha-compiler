FROM ghcr.io/rust-lang/rust:nightly-alpine as builder
WORKDIR /usr/src/app
RUN apk add --no-cache build-base
COPY Cargo-docker.toml ./Cargo.toml
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release

COPY ./src/*.rs ./src/
COPY ./build.rs .
COPY ./src/*.lalrpop ./src/
COPY Cargo-docker.toml ./Cargo.toml

RUN cargo build --release

FROM alpine:latest
RUN mkdir /var/rinha
WORKDIR /var/rinha
COPY --from=builder /usr/src/app/target/release/lambda_rinha /var/rinha/lambda_rinha
CMD ["/var/rinha/lambda_rinha", "/var/rinha/source.rinha.json"]
