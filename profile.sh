# Run `cargo bench --no-run` and copy the target paths we need to profile

# Paste the paths here and keep only one path uncommented
# BENCH="target/release/deps/partition-e14de47a1b689ff8"
# BENCH="target/release/deps/rnn_search-9bfb0d2a7b7e86b1"
# BENCH="target/release/deps/knn_search-73f79a0a52d5519d"
BENCH="target/release/examples/search_tiny"

# Run this command using `bash profile.sh` to profile using valgrind
valgrind --tool=callgrind \
         --dump-instr=yes \
         --collect-jumps=yes \
         --simulate-cache=yes \
         $BENCH --release --profile-time 10

# Run `kcachegrind` to open the profile data saved be valgrind
