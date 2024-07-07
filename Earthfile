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

ENV RYE_HOME="/opt/rye"
RUN curl -sSf https://rye.astral.sh/get | RYE_INSTALL_OPTION="--yes" bash
ENV PATH="${RYE_HOME}/shims:${PATH}"

# This target prepares the recipe.json file for the build stage.
chef-prepare:
    COPY --dir crates pypi .
    COPY Cargo.toml .
    RUN cargo chef prepare
    SAVE ARTIFACT recipe.json

# This target uses the recipe.json file to build a cache of the dependencies.
chef-cook:
    COPY +chef-prepare/recipe.json ./
    RUN cargo chef cook --release
    COPY Cargo.toml pyproject.toml requirements.lock requirements-dev.lock ruff.toml rustfmt.toml .
    # TODO: Replace with recursive globbing, blocked on https://github.com/earthly/earthly/issues/1230
    COPY --dir crates .
    COPY --dir pypi .
    RUN rye sync --no-lock

# This target builds the project using the cached dependencies.
build:
    FROM +chef-cook
    RUN cargo build --release
    RUN rye build --all --out target/release/pypi
    SAVE ARTIFACT target/release AS LOCAL ./target/

# This target formats the project.
fmt:
    FROM +chef-cook
    RUN cargo fmt --all -- --check && rye fmt --all --check

# This target lints the project.
lint:
    FROM +chef-cook
    RUN cargo clippy --all-targets --all-features
    RUN rye lint --all

# Apply any automated fixes.
fix:
    FROM +chef-cook
    RUN cargo fmt --all
    RUN rye fmt --all
    RUN cargo clippy --fix --allow-no-vcs
    RUN rye lint --fix
    SAVE ARTIFACT crates AS LOCAL ./
    SAVE ARTIFACT pypi AS LOCAL ./

# This target runs the tests.
test:
    FROM +chef-cook
    RUN cargo test --release --lib --bins --examples --tests --all-features
    # TODO: switch to --all, blocked on https://github.com/astral-sh/rye/issues/853
    RUN rye test --package abd-distances

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
