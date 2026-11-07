#!/usr/bin/env bash
# Build the Apptainer image rootless, politely (shared machine): nice 19, idle I/O, at most 4 cores,
# build cache and temp files under $RERUN_ROOT/cache instead of the shared /tmp.
#
#   env/build_image.sh [OUT.sif]      default: $RERUN_ROOT/images/repattack-v2.sif
#
# Uses `apptainer` from PATH, else $RERUN_ROOT/tools/apptainer/bin/apptainer (unprivileged install:
# extract the official release .deb with `dpkg-deb -x`, move usr/* up one level; see docs/REPRODUCE.md).
set -euo pipefail
RERUN_ROOT=${RERUN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)/rerun}
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-$RERUN_ROOT/images/repattack-v2.sif}
APPTAINER=$(command -v apptainer || echo "$RERUN_ROOT/tools/apptainer/bin/apptainer")
export APPTAINER_CACHEDIR=$RERUN_ROOT/cache/apptainer/cache APPTAINER_TMPDIR=$RERUN_ROOT/cache/apptainer/tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$(dirname "$OUT")"
CPUS=${RERUN_CPUS:-$(nproc --all | awk '{print $1-4 "-" $1-1}')}
cd "$REPO"
taskset -c "$CPUS" nice -n 19 ionice -c3 "$APPTAINER" build --fakeroot --ignore-subuid "$OUT" env/apptainer.def
sha256sum "$OUT" | tee "$OUT.sha256"
echo "built from $(git -C "$REPO" rev-parse HEAD)$(git -C "$REPO" diff --quiet HEAD -- env || echo ' (env/ modified)')" > "$OUT.commit"
