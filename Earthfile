VERSION 0.8
FROM rust:latest
WORKDIR /usr/local/src

ENV DEBIAN_FRONTEND noninteractive

# The cache directive creates a buildkit cache mount point. This allows us to cache the dependencies between builds.
CACHE ./target/

# Add any additional rustup components here. They are available in all targets.
RUN rustup component add \
    rustfmt \
    clippy

# Add any additional packages here. They are available in all targets.
RUN cargo install \
    cargo-chef \
    jaq

RUN cargo install maturin --locked

# This target prepares the recipe.json file for the build stage.
chef-prepare:
    COPY --dir crates .
    COPY --dir python .
    COPY Cargo.toml .
    RUN cargo chef prepare
    SAVE ARTIFACT recipe.json

# This target uses the recipe.json file to build a cache of the dependencies.
chef-cook:
    COPY +chef-prepare/recipe.json ./
    RUN cargo chef cook --release
    COPY --dir crates .
    COPY --dir python .
    COPY Cargo.toml .

# This target builds the project using the cached dependencies.
build:
    FROM +chef-cook
    RUN cargo build --release
    SAVE ARTIFACT target/release AS LOCAL ./target/

# This target formats the project.
fmt:
    FROM +chef-cook
    RUN cargo fmt --all
    SAVE ARTIFACT crates AS LOCAL ./
    SAVE ARTIFACT python AS LOCAL ./
    RUN cargo fmt --all -- --check

# This target lints the project.
clippy:
    FROM +fmt
    RUN cargo clippy --all-targets --all-features

# This target runs the tests.
test:
    FROM +fmt
    RUN cargo test --release --lib --bins --examples --tests --all-features

pytest:
    ARG --required PKG
    FROM ghcr.io/pyo3/maturin:latest
    COPY --dir crates .
    COPY --dir python .
    COPY Cargo.toml .
    RUN python -m venv .venv && . .venv/bin/activate && pip install --upgrade pip
    WORKDIR /usr/local/src/python/$PKG
    RUN . ../../.venv/bin/activate && maturin develop --release --strip --extras=dev
    RUN . ../../.venv/bin/activate && python -m pytest -v

pybench:
    ARG --required PKG
    FROM +pytest
    WORKDIR /usr/local/src/python/$PKG
    RUN . ../../.venv/bin/activate && python -m richbench benches --markdown

# This target runs the tests on aarch64, it can be expanded to run tests on additional platforms, but it is SLOW.
cross-test:
    FROM +fmt
    RUN cargo install cross --git  https://github.com/cross-rs/cross
    WITH DOCKER
        RUN cross test --target aarch64-unknown-linux-gnu --all-features
    END

# This target runs the benchmarks.
bench:
    FROM +fmt
    RUN cargo bench --all-features
