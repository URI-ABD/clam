# Run `cargo bench --no-run` and copy the target paths we need to profile

# Paste the path here
BENCH="target/release/deps/rnn_search-59b89a1514a8c2aa"

# Run this command using `bash profile.sh` to profile using samply
samply record ./$BENCH
