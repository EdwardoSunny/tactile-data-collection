# tactile-data-collection

Teleop + data-collection pipeline for the xArm + 2-finger tactile rig. Two steps end-to-end: **collect** raw recordings from the rig, then **render overlays** to produce training-ready data. Downstream training frameworks (ACT, diffusion_policy, openpi) each have their own converter that reads the overlay output.

For implementation details — thread model, recorder semantics, the tactile pipeline, the on-disk schema in full — see [`CLAUDE.md`](CLAUDE.md).

---

## Pipeline

```
   collect_and_render.sh         ← single entry point for a full session
       │
       ├─► collect_with_home.py --record --viz
       │       │
       │       ▼
       │   teleop_data.zarr      ← RAW: state, tactile, joint_angles,
       │                            grip_pos, un-overlaid camera frames.
       │                            APPENDED to across sessions.
       │
       └─► scripts/render_overlays.py     (auto-run after collect exits)
               │
               ▼
           teleop_data_overlay.zarr   ← TRAINING-READY:
                                        • all raw fields, copied through
                                        • 6 overlay-rendered image variants
                                          per camera per frame
                                        • dataset-wide tactile normalization
                                        REGENERATED FROM SCRATCH each run.
```

The raw zarr is the source of truth; the overlay zarr is a deterministic, throw-away-able rebuild from it. Re-running `collect_and_render.sh` after collecting more episodes appends to the raw zarr and regenerates the overlay zarr against the whole (newly grown) dataset.

---

## 1. Collect

```bash
./collect_and_render.sh                         # standard session
./collect_and_render.sh --safety-threshold 1800 # override + forward flags
RAW_ZARR=foo.zarr OVERLAY_ZARR=foo_overlay.zarr ./collect_and_render.sh
```

In-session controls:

| key | action |
|---|---|
| **phone button** | start/stop an episode (3 s cooldown; robot re-homes before each new episode) |
| **Backspace** | discard the most recently completed episode |
| **Ctrl+C** | quit and flush to disk |

After the script exits:
- `teleop_data.zarr` — raw data, appended to (never overwritten).
- `teleop_data_overlay.zarr` — overlay-augmented data, wiped + rebuilt from the current raw zarr.

The render step always runs, even if `collect_with_home.py` exited non-zero (Ctrl+C produces a non-zero status on some shells and we still want the overlay to refresh).

See `CLAUDE.md > Running things` for the full `collect_with_home.py` flag list (`--no-tactile`, `--viz-mode`, etc.) and the on-disk schema.

---

## 2. Render overlays (only if you need to rebuild without recording)

`render_overlays.py` is auto-run by `collect_and_render.sh`. Run it by hand when you want to re-render after tweaking normalization or arrow styling, without collecting new data:

```bash
python scripts/render_overlays.py                                     # uses defaults
python scripts/render_overlays.py raw.zarr overlay.zarr               # explicit paths
```

What it produces in the output zarr, per camera index `i` ∈ {0=agent, 1=wrist}:

| key | shape | what |
|---|---|---|
| `/data/img_{i}` | (N, 224, 224, 3) | un-overlaid frame, copied from raw |
| `/data/img_{i}_points9_arrow` | (N, 224, 224, 3) | 9 arrows per finger, one per cell |
| `/data/img_{i}_points1_arrow` | (N, 224, 224, 3) | 1 aggregated arrow per finger |
| `/data/img_{i}_points1_contact_spatial` | (N, 224, 224, 3) | binary contact dot, projected onto finger |
| `/data/img_{i}_points9_color_spatial` | (N, 224, 224, 3) | per-cell colored dots, projected onto finger |
| `/data/img_{i}_points1_contact_flat` | (N, 224, 224, 3) | binary contact dot, fixed image corner |
| `/data/img_{i}_points9_color_flat` | (N, 224, 224, 3) | per-cell colored dots, fixed image corner |

Plus the raw fields (`state`, `joint_angles`, `grip_pos`, `tactile*`, `n_contacts`) and `/meta` (episode_ends, tactile_baseline) — all copied verbatim from the raw zarr.

### Normalization and arrow styling

`render_overlays.py` computes tactile normalization **from the dataset itself**, not from session-start baselines or hardware calibration. Specifically:

- **per-cell, per-axis offset** = median across all frames in the input zarr (robust idle estimate).
- **per-finger XY scale** = 99th percentile of `|xy − offset_xy|` across all frames/cells for that finger.
- **per-finger Z scale** = 99th percentile of `|z − offset_z|`.
- **noise deadband** = 0.12 — per-cell normalized vectors with L2 magnitude below this draw nothing (kills idle jitter so empty-grasp frames render with no arrows).

The arrows visually saturate against **this task's own contact range**, so cube-light and tube-light contacts both render as small arrows even though their raw force counts differ. The constants live at the top of `scripts/render_overlays.py`:

```python
DATASET_PERCENTILE   = 99.0       # bigger = arrows saturate later, less noise amplification
NOISE_DEADBAND       = 0.12       # bigger = more idle jitter killed, weaker contacts hidden
BOLD_ARROW_THICKNESS = 8          # px
BOLD_DOT_SIZE        = 22         # px
```

Per-mode arrow scales live in `environment/tactile_overlay.MODES` — adjust `arrow_length_scale` per variant to retune visual length.

---

## 3. Preview an episode as MP4

To sanity-check a rendered overlay before committing to a training run:

```bash
# defaults: cube episode 0, points9_arrow, agent + wrist side-by-side, 10 fps
python scripts/episode_to_mp4.py

# tube, episode 3, single-arrow variant, single camera, output path
python scripts/episode_to_mp4.py /data/edward/teleop_data_tube_overlay.zarr \
    --episode 3 --mode points1_arrow --camera 0 --out tube_ep3.mp4
```

Available `--mode`: `raw`, `points9_arrow`, `points1_arrow`, `points1_contact_spatial`, `points9_color_spatial`, `points1_contact_flat`, `points9_color_flat`. The script just reads pre-rendered frames from the overlay zarr and packs them into an mp4 — what you see is what training will see (modulo mp4 codec quantization, which is negligible at normal viewing distance).

---

## 4. Train (in the downstream repo)

The overlay zarr is the canonical training input for this repo. Each downstream framework owns its own converter that reads it:

- **ACT** — `/data/edward/act/convert_zarr_to_act.py` (writes per-episode HDF5 files)
- **diffusion_policy** — `/data/edward/diffusion_policy/diffusion_policy/scripts/tactile_xarm_conversion.py` (writes a ReplayBuffer-compatible zarr) + `diffusion_policy/dataset/xarm_image_dataset.py` (the Dataset class)
- **openpi** — `/data/edward/openpi/examples/xarm/convert_zarr_to_lerobot.py` (writes a LeRobot dataset, with `--mode {raw,points9_arrow,...}` to pick the overlay variant baked into the training images)

Each converter expects to read from `teleop_data_overlay.zarr` and handles the per-framework state/action conventions (10-dim 6D rotation, 7-dim axis-angle delta, per-episode HDF5, etc.) internally. See the docstring at the top of each script for usage.

---

## Other utilities

```bash
# Extract a stripped canonical-raw zarr from a source zarr that may have
# legacy or extra fields. Validates required fields are present.
python scripts/extract_raw.py SRC.zarr DST.zarr

# Inline browser for any zarr (raw or overlay). Hotkeys r/a/g/p/b/9 etc.
# switch the displayed image variant.
python scripts/dataset_viewer.py --data DST.zarr --mode points9_arrow

# Misc: count_data.py, combine_zarrs.py, repack_zarr.py, tune_safety.py,
# tune_wrist_tactile.py, print_episode_states.py — see each file's
# header docstring.
```

---

## Quick reference

```bash
# COLLECT (auto-renders overlay after)
./collect_and_render.sh

# REBUILD overlay only (e.g. after tweaking normalization or arrow style)
python scripts/render_overlays.py /data/edward/teleop_data.zarr /data/edward/teleop_data_overlay.zarr

# PREVIEW one episode as mp4
python scripts/episode_to_mp4.py /data/edward/teleop_data_cube_overlay.zarr \
    --episode 0 --mode points9_arrow --out preview.mp4

# EXPORT to a downstream training repo (in its own repo, not here)
python /data/edward/act/convert_zarr_to_act.py /data/edward/teleop_data_overlay.zarr --out <dir>
python -m diffusion_policy.scripts.tactile_xarm_conversion /data/edward/teleop_data_overlay.zarr <dst.zarr>   # from /data/edward/diffusion_policy
uv run examples/xarm/convert_zarr_to_lerobot.py --zarr /data/edward/teleop_data_overlay.zarr --mode points9_arrow  # from /data/edward/openpi
```
