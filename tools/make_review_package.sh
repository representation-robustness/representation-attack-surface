#!/usr/bin/env bash
# Build an anonymized, self-contained review package (code + all derived results), for double-blind review.
#
#   tools/make_review_package.sh OUT_DIR
#
# OUT_DIR/README.md   how to use the package
# OUT_DIR/repo/       `git archive HEAD` of this repository
# OUT_DIR/results/    results/{tables,figures,stats,metrics,attacks,audit,preds} from $RERUN_ROOT
# OUT_DIR/runs/       per-job manifests;  OUT_DIR/logs/  per-job logs
# Absolute paths, the host name and user names are replaced by placeholders in the copies. The package is
# then scanned for identifying strings (IDENTIFYING env var, a |-separated regex), and the build fails on a hit.
# Nothing is uploaded: the package is written locally.
set -euo pipefail
OUT=${1:?usage: $0 OUT_DIR}
REPO=$(cd "$(dirname "$0")/.." && pwd)
RERUN_ROOT=${RERUN_ROOT:-$(cd "$REPO/.." && pwd)/rerun}
IDENTIFYING=${IDENTIFYING:-}
[[ -e $OUT ]] && { echo "$OUT exists; choose a new directory"; exit 1; }
mkdir -p "$OUT/repo"
cat > "$OUT/README.md" <<'EOR'
# Review package

- `repo/`: the artifact repository (start with `repo/README.md` and `repo/docs/PAPER_MAP.md`).
- `results/`: the reported results, including `results/preds/`, the per-sample predictions of every run
  (Devign, DiverseVul, Big-Vul, PrimeVul and DeepWukong).
- `runs/manifests/`, `logs/`: per-job provenance and logs (paths and host names replaced by placeholders).

To regenerate the tables without GPUs, follow "Rebuild every table without GPUs" in `repo/README.md`, but instead of
the `predictions` stage copy the predictions into place: `mkdir -p $RERUN_ROOT/results && cp -r results/preds
$RERUN_ROOT/results/`. With the DeepWukong predictions present, the `results` stage also needs the DeepWukong data
(`./reproduce.sh deepwukong`); without it, remove `$RERUN_ROOT/results/preds/deepwukong` first.
EOR
git -C "$REPO" archive HEAD | tar -x -C "$OUT/repo"
rm -f "$OUT/repo/docs/RERUN_JOURNAL.md"   # only present in a development checkout
mkdir -p "$OUT/results"
for d in tables figures stats metrics attacks audit preds; do
  if [[ -d $RERUN_ROOT/results/$d ]]; then cp -r "$RERUN_ROOT/results/$d" "$OUT/results/"; fi
done
# figures: the paper figures as committed in the repository (RERUN_ROOT/results/figures also holds working files)
rm -rf "$OUT/results/figures" && cp -r "$OUT/repo/results/figures" "$OUT/results/figures"
# tables as received from the PrimeVul workspace (superseded by the regenerated tables; internal record only)
rm -rf "$OUT/results/tables/primevul_source"
# reconstruction of the earlier DeepWukong setup (internal check, not reported)
rm -f "$OUT/results/tables/"dwk_legacy_reconstruction*
mkdir -p "$OUT/runs" && cp -r "$RERUN_ROOT/runs/manifests" "$OUT/runs/"
cp -r "$RERUN_ROOT/logs" "$OUT/logs"
rm -rf "$OUT/logs/"*backup*
rm -f "$OUT/logs/"dwk_legacy_reconstruction*

scrub() {  # placeholders for absolute paths, host and user in text files (gz files are rewritten in place)
  local host user; host=$(hostname -f 2>/dev/null || hostname); user=$(id -un)
  local expr="s#$RERUN_ROOT#\$RERUN_ROOT#g; s#$REPO#\$REPO#g; s#$HOME#\$HOME#g; s#$host#HOST#g; s#\\b$user\\b#USER#g"
  find "$OUT" -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.log' -o -name '*.md' -o -name '*.txt' -o -name '*.csv' -o -name '*.tex' -o -name '*.sh' -o -name '*.py' -o -name '*.yaml' \) \
       -print0 | xargs -0 -r sed -i "$expr"
  find "$OUT" -type f -name '*.gz' -print0 | while IFS= read -r -d '' f; do
    zcat "$f" | sed "$expr" | gzip -n > "$f.tmp" && mv "$f.tmp" "$f"
  done
}
scrub

pat="$(id -un)|$(hostname)${IDENTIFYING:+|$IDENTIFYING}"
if grep -r -I -l -i -E "$pat" "$OUT" | head -5 | grep -q .; then
  echo "identifying strings remain in:"; grep -r -I -l -i -E "$pat" "$OUT" | head -20; exit 1
fi
if find "$OUT" -name '*.gz' -print0 | xargs -0 -r zgrep -l -i -E "$pat" | head -1 | grep -q .; then
  echo "identifying strings remain in compressed files"; exit 1
fi
du -sh "$OUT"; echo "review package ready: $OUT (nothing uploaded)"
