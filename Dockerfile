FROM rust:1.40-stretch

# Dependancies.
RUN apt-get update --fix-missing \
    && apt-get install -y \
        cmake \
        libssl-dev \
        pkg-config \
        zlib1g-dev

# Nightly toolchain
RUN rustup toolchain install nightly

# Copy source.
WORKDIR /usr/src/distance
COPY . .

# Default command runs tests.
CMD ["cargo", "+nightly", "test"]
