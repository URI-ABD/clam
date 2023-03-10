# Run `cargo bench --no-run` and copy the target paths we need to profile

# Paste the paths here and keep only one path uncommented
# BENCH="target/release/examples/triangle"
# BENCH="target/release/examples/tetrahedron"
BENCH="target/release/examples/search_tiny"

# Run this command using `bash profile.sh` to profile using valgrind
valgrind --tool=callgrind \
         --dump-instr=yes \
         --collect-jumps=yes \
         --simulate-cache=yes \
         $BENCH --release --profile-time 10

# Run `kcachegrind` to open the profile data saved be valgrind
