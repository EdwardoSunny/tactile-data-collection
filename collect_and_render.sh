#!/usr/bin/env bash
#
# Run a full data-collection session, then regenerate the overlay-augmented
# dataset from the (newly-grown) raw zarr.
#
# Forwards any extra arguments to collect_with_home.py — e.g.:
#   ./collect_and_render.sh --reset-duration 4 --safety-threshold 1800
#
# The render step ALWAYS runs after collect_with_home.py exits, whether by
# normal Ctrl+C or by error. We deliberately don't gate on a clean exit
# because Ctrl+C produces a nonzero status on some shells and we still want
# the overlay to refresh.
#
# Paths are relative to the current working directory (matches where
# collect_with_home.py writes teleop_data.zarr).

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RAW_ZARR="${RAW_ZARR:-teleop_data.zarr}"
OVERLAY_ZARR="${OVERLAY_ZARR:-teleop_data_overlay.zarr}"

echo "=========================================================="
echo "  Step 1/2: collect_with_home.py --record --viz  $*"
echo "  Raw zarr     : $RAW_ZARR  (appended to)"
echo "  Overlay zarr : $OVERLAY_ZARR  (regenerated after collect)"
echo "=========================================================="

# Note: don't `set -e` around this — Ctrl+C exits with non-zero status,
# and we still want the render to run on the data that was captured.
python "$REPO_ROOT/collect_with_home.py" --record --viz "$@"
COLLECT_STATUS=$?

echo
echo "=========================================================="
echo "  Step 2/2: render_overlays.py $RAW_ZARR -> $OVERLAY_ZARR"
echo "=========================================================="

if [ ! -d "$RAW_ZARR" ]; then
    echo "  [skip] $RAW_ZARR does not exist — nothing to render."
    exit "$COLLECT_STATUS"
fi

python "$REPO_ROOT/scripts/render_overlays.py" "$RAW_ZARR" "$OVERLAY_ZARR"
RENDER_STATUS=$?

# Surface whichever step failed; prefer the collect exit code when both are
# zero so $? reflects "did the data-collection step finish cleanly".
if [ "$COLLECT_STATUS" -ne 0 ]; then
    exit "$COLLECT_STATUS"
fi
exit "$RENDER_STATUS"
