VERSION 0.7
FROM ubuntu:latest
WORKDIR /usr/local/src

ENV DEBIAN_FRONTEND noninteractive

# Run updates
RUN apt-get update && \
    apt-get install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    build-essential \
    curl \
    git \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3.9-venv \
    python3-pip && \
    apt-get update

# Install rust
RUN curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

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

# Create a virtual environment for the python dependencies.
RUN python3.9 -m venv .venv
RUN . .venv/bin/activate && pip install --upgrade pip

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
    FROM +fmt
    WORKDIR /usr/local/src/python/distances
    RUN maturin develop --release --strip --extras=dev
    RUN . ../../.venv/bin/activate && python -m pytest -v

pybench:
    FROM +pytest
    WORKDIR /usr/local/src/python/distances
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
