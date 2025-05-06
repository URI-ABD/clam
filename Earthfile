VERSION 0.8
FROM rust:latest
WORKDIR /usr/local/src

ENV DEBIAN_FRONTEND noninteractive

# The cache directive creates a buildkit cache mount point. This allows us to cache the dependencies between builds.
CACHE --persist ./target/
# TODO: https://github.com/astral-sh/rye/issues/868
# CACHE --persist ./.venv/

# Add any additional rustup components here. They are available in all targets.
RUN rustup component add \
    rustfmt \
    clippy

# Add any additional packages here. They are available in all targets.
RUN cargo install \
    cargo-chef \
    jaq \
    maturin \
    --locked

RUN apt-get update
RUN apt-get install build-essential
RUN apt-get install -y libhdf5-dev

# Install uv and add it to the path.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin/:$PATH"

# This target prepares the recipe.json file for the build stage.
chef-prepare:
    COPY --dir benches clts crates pypi .
    COPY Cargo.toml .
    RUN cargo chef prepare
    SAVE ARTIFACT recipe.json

# This target uses the recipe.json file to build a cache of the dependencies.
chef-cook:
    COPY +chef-prepare/recipe.json ./
    RUN cargo chef cook --release
    COPY Cargo.toml pyproject.toml ruff.toml rustfmt.toml .
    # TODO: Replace with recursive globbing, blocked on https://github.com/earthly/earthly/issues/1230
    COPY --dir benches .
    COPY --dir clts .
    COPY --dir crates .
    COPY --dir pypi .
    RUN uv sync --all-extras --all-packages --upgrade

# This target builds the project using the cached dependencies.
build:
    FROM +chef-cook
    RUN cargo build --release
    RUN uv build --all --out-dir target/release/pypi
    SAVE ARTIFACT target/release AS LOCAL ./target/

# This target formats the project.
fmt:
    FROM +chef-cook
    RUN cargo fmt --all -- --check && uv run ruff check

# This target lints the project.
lint:
    FROM +chef-cook
    RUN cargo clippy --all-targets --all-features
    RUN uv run ruff check

# Apply any automated fixes.
fix:
    FROM +chef-cook
    RUN cargo fmt --all --all-features
    RUN uv run ruff format
    RUN cargo clippy --fix --allow-no-vcs
    RUN uv run ruff check --fix
    SAVE ARTIFACT benches AS LOCAL ./
    SAVE ARTIFACT clts AS LOCAL ./
    SAVE ARTIFACT crates AS LOCAL ./
    SAVE ARTIFACT pypi AS LOCAL ./

# This target runs the tests.
test:
    FROM +chef-cook
    RUN cargo test -r -p abd-clam --all-features -p distances -p symagen
    RUN uv run pytest

# This target runs the tests on aarch64, it can be expanded to run tests on additional platforms, but it is SLOW.
cross-test:
    FROM +chef-cook
    RUN cargo install cross --git  https://github.com/cross-rs/cross
    WITH DOCKER
        RUN cross test --target aarch64-unknown-linux-gnu --all-features
    END

# This target runs the benchmarks.
bench:
    FROM +chef-cook
    # TODO: This is currently broken.
    # RUN cargo bench --all-features
    FOR project IN $(cd pypi && ls -d */ | sed '/\./d;s%/$%%')
        BUILD "./pypi+bench" --DIR=$project
    END

# Helper target to check the most important other targets.
all:
    BUILD +build
    BUILD +fmt
    BUILD +lint
    BUILD +test
