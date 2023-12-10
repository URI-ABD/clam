VERSION 0.7
FROM rust:latest
WORKDIR /usr/local/src
CACHE ./target/

RUN rustup component add \
    rustfmt \
    clippy

RUN cargo install \
    cargo-chef \
    jaq

chef-prepare:
    COPY --dir crates Cargo.toml .
    RUN cargo chef prepare
    SAVE ARTIFACT recipe.json

chef-cook:
    COPY +chef-prepare/recipe.json ./
    RUN cargo chef cook --release
    COPY --dir crates Cargo.toml .

build:
    FROM +chef-cook
    RUN cargo build --release
    SAVE ARTIFACT target/release AS LOCAL ./target/

fmt:
    FROM +chef-cook
    RUN cargo fmt --all
    SAVE ARTIFACT crates AS LOCAL ./
    RUN cargo fmt --all -- --check

clippy:
    FROM +fmt
    RUN cargo clippy --all-targets --all-features

test:
    FROM +fmt
    RUN cargo test --lib --bins --examples --tests --all-features

cross-test:
    FROM +fmt
    RUN cargo install cross --git  https://github.com/cross-rs/cross
    WITH DOCKER
        RUN cross test --target aarch64-unknown-linux-gnu --all-features
    END

bench:
    FROM +fmt
    RUN cargo bench --all-features
