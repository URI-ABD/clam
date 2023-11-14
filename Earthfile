VERSION 0.7
FROM rust:latest
WORKDIR /usr/local/src

RUN rustup component add \
    rustfmt \
    clippy

tools:
    RUN cargo install \
        cargo-chef 

chef-prepare:
    FROM +tools
    CACHE ./target/
    COPY --dir crates Cargo.lock Cargo.toml .
    RUN cargo chef prepare
    SAVE ARTIFACT recipe.json

build-cache:
    FROM +tools
    CACHE ./target/
    COPY +chef-prepare/recipe.json ./
    CACHE target
    RUN cargo chef cook --release
    COPY --dir crates Cargo.lock Cargo.toml .

build:
    FROM +build-cache
    CACHE ./target/
    RUN cargo build --release
    SAVE ARTIFACT target/release AS LOCAL ./target/

fmt:
    FROM +build-cache
    CACHE ./target/
    RUN cargo fmt --all
    SAVE ARTIFACT crates AS LOCAL ./
    RUN cargo fmt --all -- --check

clippy:
    FROM +fmt
    CACHE ./target/
    RUN cargo clippy --all-targets --all-features

test:
    FROM +fmt
    CACHE ./target/
    RUN cargo test --all-targets --all-features
