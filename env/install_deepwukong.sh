#!/usr/bin/env bash
# DeepWukong environment (separate from the main one: PyTorch Lightning 2.x, JDK 11 for its bundled Joern).
#   env/install_deepwukong.sh            -> $RERUN_ROOT/venvs/dwk, $RERUN_ROOT/tools/jdk11
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
RERUN_ROOT=${RERUN_ROOT:-$(cd "$REPO/.." && pwd)/rerun}
JDK_URL=https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.25%2B9/OpenJDK11U-jdk_x64_linux_hotspot_11.0.25_9.tar.gz
JDK_SHA=191baa2e052627614022171400a917d28f0987dc54da48aaf07b06f552bb9884

if [[ ! -x $RERUN_ROOT/tools/jdk11/bin/java ]]; then
  tmp=$(mktemp -d); curl -fsSL -o "$tmp/jdk.tgz" "$JDK_URL"
  echo "$JDK_SHA  $tmp/jdk.tgz" | sha256sum -c -
  mkdir -p "$RERUN_ROOT/tools/jdk11" && tar -xzf "$tmp/jdk.tgz" -C "$RERUN_ROOT/tools/jdk11" --strip-components=1
  rm -rf "$tmp"
fi
if [[ ! -x $RERUN_ROOT/venvs/dwk/bin/python ]]; then
  python3.10 -m venv "$RERUN_ROOT/venvs/dwk"
  "$RERUN_ROOT/venvs/dwk/bin/pip" install -q pip==24.2 setuptools==75.1.0 wheel==0.44.0
  "$RERUN_ROOT/venvs/dwk/bin/pip" install -q -r "$REPO/env/requirements-deepwukong.lock.txt" \
      --extra-index-url https://download.pytorch.org/whl/cu121
fi
"$RERUN_ROOT/tools/jdk11/bin/java" -version 2>&1 | head -1
"$RERUN_ROOT/venvs/dwk/bin/python" -c "import torch, pytorch_lightning; print('torch', torch.__version__, 'lightning', pytorch_lightning.__version__)"
