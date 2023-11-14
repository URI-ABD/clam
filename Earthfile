VERSION 0.7
FROM rust:latest
WORKDIR /usr/local/src

tools:
    RUN cargo install \
        cargo-chef 

chef-prepare:
    FROM +tools
    COPY --dir crates Cargo.lock Cargo.toml .
    RUN cargo chef prepare
    SAVE ARTIFACT recipe.json

build-cache:
    FROM +tools
    COPY +chef-prepare/recipe.json ./
    CACHE target
    RUN cargo chef cook --release

build:
    FROM +build-cache
    COPY --dir crates Cargo.lock Cargo.toml .
    RUN cargo build --release
    SAVE ARTIFACT target/release 
