FROM rust:1.54

# Dependancies.
RUN apt-get update --fix-missing \
    && apt-get install -y \
        cmake \
        libssl-dev \
        pkg-config \
        zlib1g-dev

# Copy source.
WORKDIR /usr/src/clam
COPY . .

RUN cargo build --release

# Default command runs tests.
CMD ["/usr/src/clam/target/release/clam"]
