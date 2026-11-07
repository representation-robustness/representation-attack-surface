#!/usr/bin/env bash
# Install the full software stack for the v2 rerun under one prefix. Shared by env/Dockerfile and
# env/apptainer.def so both images are built by the same steps; every download is version-pinned
# and checksummed against env/upstream.lock.
#
#   env/install_env.sh PREFIX        (needs: python3.10 + venv, curl, unzip, git, gcc, g++)
#
# Produces PREFIX/{jdk, joern-v4.0.655, venv, upstream/*}. Point the pipeline at it with
#   RERUN_PY=PREFIX/venv/bin/python RERUN_JDK=PREFIX/jdk RERUN_JOERN=PREFIX/joern-v4.0.655 RERUN_UPSTREAM=PREFIX/upstream
set -euo pipefail
PREFIX=$1
ENVDIR=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$PREFIX"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Java: Temurin 21.0.12+8
JDK_URL="https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.12%2B8/OpenJDK21U-jdk_x64_linux_hotspot_21.0.12_8.tar.gz"
JDK_SHA256=$(grep '^jdk_sha256:' "$ENVDIR/upstream.lock" | awk '{print $2}')
curl -fsSL -o "$TMP/jdk.tgz" "$JDK_URL"
echo "$JDK_SHA256  $TMP/jdk.tgz" | sha256sum -c -
mkdir -p "$PREFIX/jdk" && tar -xzf "$TMP/jdk.tgz" -C "$PREFIX/jdk" --strip-components=1

# Joern v4.0.655 (sha512 from the release, cross-checked with the lock)
JOERN_URL=https://github.com/joernio/joern/releases/download/v4.0.655/joern-cli-linux-x86_64.zip
JOERN_SHA512=$(grep '^joern_sha512:' "$ENVDIR/upstream.lock" | awk '{print $2}')
curl -fsSL -o "$TMP/joern.zip" "$JOERN_URL"
echo "$JOERN_SHA512  $TMP/joern.zip" | sha512sum -c -
mkdir -p "$PREFIX/joern-v4.0.655" && unzip -q "$TMP/joern.zip" -d "$PREFIX/joern-v4.0.655"

# Python: the exact frozen set of the reference venv (CUDA 12.1 wheels come from pip, not the base image)
python3.10 -m venv "$PREFIX/venv"
"$PREFIX/venv/bin/pip" install --no-cache-dir --upgrade pip==24.2 setuptools==75.1.0 wheel==0.44.0
"$PREFIX/venv/bin/pip" install --no-cache-dir -r "$ENVDIR/requirements-main.lock.txt" \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

# Upstream code at pinned commits (shallow fetch of exactly that commit)
mkdir -p "$PREFIX/upstream"
grep '^git:' "$ENVDIR/upstream.lock" | while read -r _ url sha; do
    name=$(basename "$url" .git)
    [[ $name == DeepWukong ]] && continue   # separate environment (see docs/REPRODUCE.md)
    d="$PREFIX/upstream/$name"
    git init -q "$d"
    git -C "$d" fetch -q --depth 1 "$url" "$sha"
    git -C "$d" -c advice.detachedHead=false checkout -q FETCH_HEAD
    [[ $(git -C "$d" rev-parse HEAD) == "$sha" ]]
done
