"""
Teleop + recording entry point.

Phone teleop drives the robot in XY/Z/RPY; on each phone-button rising-edge
(with a 3 s cooldown), we either start a new recording episode — first
homing the robot smoothly and re-anchoring the phone's AR frame to the
freshly homed pose — or stop the current episode.

Tactile (two A31301 boards over USB serial) is optional. When enabled it
drives two things:
  - Gripper safety: XArm.step_abs clamps a closing grasp command to the
    previous value when the contact metric exceeds threshold (or readings
    are stale). Wired through environment/xarm_controller.py.
  - Camera overlay: per-cell force arrows drawn on the agent + wrist
    images before they are downsized to 224x224 and recorded.

Quit with Ctrl+C; pynput keystroke listening is intentionally disabled so
the script works headless over SSH.
"""
import argparse
import os
import time

import cv2
import numpy as np

import tactile_config as tc
from environment import tactile_overlay
from environment.phone import Phone
from environment.tactile import TactileConfig, TactileSensors
from recorder import DatasetRecorder
from tasks.simple_task import Simple_Task
from threads import PhoneReadThread, RecordingThread


def _capture_tactile_baseline(tactile, duration_sec=1.5):
    """Sample tactile frames while gripper is open at home, average per-cell.

    Returns a (2, 9, 3) baseline array (xyz per cell per finger), or None if
    no usable samples arrived. Caller installs this on tactile_overlay so
    the per-cell static field is subtracted from the squeeze magnitude.
    """
    if tactile is None:
        return None
    samples = []
    t_end = time.time() + duration_sec
    while time.time() < t_end:
        states = tactile.get_latest()
        # Only count samples from frames where both fingers have published
        # at least once (host_timestamp > 0).
        if all(s.get("host_timestamp", 0.0) > 0 for s in states[:2]):
            xyz_L = np.asarray(states[0]["xyz"], dtype=np.float32)
            xyz_R = np.asarray(states[1]["xyz"], dtype=np.float32)
            samples.append(np.stack([xyz_L, xyz_R]))
        time.sleep(0.05)
    if not samples:
        return None
    return np.mean(np.stack(samples), axis=0).astype(np.float32)


def _load_agent_transform(npz_path):
    """Load (serial, trc 3x4) from the converted .npz. Returns (None, None) on failure."""
    if not os.path.isfile(npz_path):
        print(f"  [warn] transforms file not found: {npz_path}  -> agent overlay disabled")
        return None, None
    try:
        d = np.load(npz_path, allow_pickle=False)
        return str(d["serial"]), np.asarray(d["trc"], dtype=np.float64)
    except Exception as e:
        print(f"  [warn] couldn't load {npz_path}: {e}  -> agent overlay disabled")
        return None, None


def _parse_args():
    p = argparse.ArgumentParser(
        description="Teleop + recording with tactile safety + overlay. "
                    "Homes the robot to a fixed start pose before each new episode."
    )
    p.add_argument("--record", action="store_true", help="Enable data recording")
    p.add_argument("--reset-duration", type=float, default=3.0,
                   help="Seconds for the smooth home motion before each episode")

    # Tactile options
    p.add_argument("--left-port", type=str, default=tc.LEFT_FINGER_PORT,
                   help="Serial port for the LEFT-finger ESP32")
    p.add_argument("--right-port", type=str, default=tc.RIGHT_FINGER_PORT,
                   help="Serial port for the RIGHT-finger ESP32")
    p.add_argument("--tactile-baud", type=int, default=tc.TACTILE_BAUD,
                   help="Baud rate for both tactile serial ports")
    p.add_argument("--no-tactile", action="store_true",
                   help="Skip tactile entirely (no safety wrapper, no overlay, "
                        "no tactile zarr columns)")
    p.add_argument("--no-overlay", action="store_true",
                   help="Record raw camera frames; still record tactile values "
                        "and apply gripper safety if tactile is enabled")
    p.add_argument("--safety-threshold", type=float, default=tc.SAFETY_THRESHOLD,
                   help="Tactile metric above this -> hold grasp (no further closing)")
    p.add_argument("--safety-metric", type=str, default=tc.SAFETY_METRIC,
                   choices=["max_abs_z", "max_norm", "sum_abs_z"],
                   help="Reduction over per-taxel xyz used to detect contact")
    p.add_argument("--viz", action="store_true",
                   help="Show live agent + wrist overlay windows during the session "
                        "(needs a DISPLAY; works without --record too).")
    p.add_argument("--viz-mode", type=str, default=tc.VISUALIZATION_MODE,
                   choices=["arrow", "grid", "point", "bar"],
                   help="What the overlay draws: arrow (single per finger; default), "
                        "grid (9 arrows per finger), point (single circle per finger), "
                        "bar (binary bottom-bars per finger).")
    p.add_argument("--no-record-video", action="store_true",
                   help="Don't write per-episode mp4 files alongside the zarr "
                        "(saved by default to teleop_data.zarr_videos/).")
    return p.parse_args()


def _build_tactile(args):
    """Open TactileSensors based on args, or return None if disabled / both ports failed."""
    if args.no_tactile:
        return None
    cfg = TactileConfig(
        ports=[args.left_port, args.right_port],
        baud=args.tactile_baud,
        safety_metric=args.safety_metric,
        safety_threshold=args.safety_threshold,
        stale_after_sec=tc.SAFETY_STALE_AFTER_SEC,
        lag_warning_ms=tc.TACTILE_LAG_WARNING_MS,
    )
    tactile = TactileSensors(cfg, names=["L", "R"])
    tactile.__enter__()
    if not tactile.any_open:
        print("  [warn] no tactile ports opened  -> continuing without tactile")
        tactile.__exit__(None, None, None)
        return None
    if not tactile.all_open:
        print("  [warn] some tactile ports failed to open; continuing with partial data")
    return tactile


def _camera_role_wiring(env):
    """Resolve agent/wrist camera serials -> camera indices, plus the agent transform."""
    cameras = env.env.cameras
    serial_to_index = {c.serial_number: c.index for c in cameras}
    intrinsics_by_serial = {c.serial_number: c.intrinsics for c in cameras}
    agent_serial_from_file, trc_agent = _load_agent_transform(tc.TRANSFORMS_NPZ_PATH)
    agent_serial = agent_serial_from_file or tc.AGENT_CAMERA_SERIAL
    wrist_serial = tc.WRIST_CAMERA_SERIAL
    intrinsics_agent = intrinsics_by_serial.get(agent_serial)
    return serial_to_index, agent_serial, wrist_serial, trc_agent, intrinsics_agent


def _print_banner(args, tactile, agent_overlay_ok, wrist_overlay_ok, frequency):
    print()
    print("=" * 60)
    print("  READY")
    print("=" * 60)
    if args.record:
        video_status = "OFF" if args.no_record_video else f"ON  (-> teleop_data.zarr_videos/)"
        print(f"  Recording   : ON ({frequency:.0f} Hz)  ->  teleop_data.zarr")
        print(f"  MP4 videos  : {video_status}")
    else:
        print(f"  Recording   : OFF  (pass --record to enable)")
    if tactile is not None:
        print(f"  Tactile     : ON  (L={args.left_port}, R={args.right_port}, "
              f"baud={args.tactile_baud})")
        print(f"  Safety      : metric={tactile.config.safety_metric}, "
              f"threshold={tactile.config.safety_threshold:.1f}")
        print(f"  Overlay     : agent={'ON' if agent_overlay_ok else 'OFF'}  "
              f"wrist={'ON' if wrist_overlay_ok else 'OFF'}  mode={args.viz_mode}")
    else:
        print(f"  Tactile     : OFF")
    print(f"  Viz windows : {'ON  (close with Ctrl+C in terminal)' if args.viz else 'OFF'}")
    print(f"  Reset pose  : pos=[400, 0, 290]  rot=[180, 0, 0]")
    print(f"  Phone btn A : start / stop episode (robot homes first)")
    print(f"  Ctrl+C      : quit and flush to disk")
    print("=" * 60)
    print()


def main():
    args = _parse_args()

    tactile = _build_tactile(args)
    try:
        env = Simple_Task(tactile=tactile)
        phone = Phone()
        env.reset(duration=args.reset_duration)
        time.sleep(args.reset_duration)

        # Let the camera + pose streams settle before reading the initial pose.
        for _ in range(20):
            obs = env.get_obs()
            time.sleep(0.1)
        phone.reset(obs["pose"])

        # Capture per-cell baseline at startup (gripper open at home, no contact).
        # Used by THREE consumers, all in delta-from-idle mode:
        #   - safety wrapper  : threshold is delta units
        #   - overlay arrows  : arrow length = |delta|
        #   - recorder /meta  : saved alongside raw /data/tactile so downstream
        #                       can compute delta whenever it wants
        # Captured unconditionally when tactile is on (NOT gated on --no-overlay)
        # because safety needs it even when the overlay is suppressed.
        baseline = None
        if tactile is not None:
            print("  Sampling tactile baseline (1.5 s, keep fingers untouched)...")
            baseline = _capture_tactile_baseline(tactile, duration_sec=1.5)
            if baseline is not None:
                # 1) overlay reads from the module-level slot
                tactile_overlay.set_baseline(baseline)
                # 2) safety wrapper reads from config.baseline
                tactile.config.baseline = baseline
                # Report per-cell |B_force| baseline so the user knows what was subtracted.
                axis = {"x": 0, "y": 1, "z": 2}[tc.FORCE_AXIS]
                bL = baseline[0, :, axis]
                bR = baseline[1, :, axis]
                print(f"  Baseline captured (subtracted from safety + overlay; raw values still saved).")
                print(f"    {tc.FORCE_AXIS}-axis L: " + " ".join(f"{v:6.0f}" for v in bL))
                print(f"    {tc.FORCE_AXIS}-axis R: " + " ".join(f"{v:6.0f}" for v in bR))
            else:
                print("  [warn] couldn't sample baseline; safety + overlay will use raw values")

        frequency = 10.0

        # Camera role / agent-extrinsics wiring (only needed if overlay or recorder)
        (serial_to_index, agent_serial, wrist_serial,
         trc_agent, intrinsics_agent) = _camera_role_wiring(env)
        agent_overlay_ok = (
            tactile is not None and trc_agent is not None
            and intrinsics_agent is not None and not args.no_overlay
        )
        wrist_overlay_ok = (tactile is not None and not args.no_overlay)
        if tactile is not None and not agent_overlay_ok and not args.no_overlay:
            print(f"  [warn] agent camera ({agent_serial}) transform or intrinsics "
                  f"unavailable  -> disabling agent overlay only")

        recorder = None
        recording_thread = None
        # Start the recording thread if EITHER --record (it writes zarr) OR --viz
        # (it computes overlays the viz windows read).
        if args.record or args.viz:
            if args.record:
                recorder = DatasetRecorder(
                    "teleop_data.zarr",
                    memory_buffer_size=5000,
                    flush_interval=2.0,
                    use_actions=False,
                    use_tactile=(tactile is not None),
                    tactile_baseline=baseline,   # saved once to /meta/tactile_baseline
                    record_videos=(not args.no_record_video),
                    video_fps=frequency,         # match the recording tick rate
                )
            recording_thread = RecordingThread(
                recorder, env, frequency,
                tactile=tactile,
                sensor_labels=["L", "R"],
                agent_serial=agent_serial,
                wrist_serial=wrist_serial,
                serial_to_index=serial_to_index,
                trc_agent=trc_agent,
                intrinsics_agent=intrinsics_agent,
                draw_overlay=(not args.no_overlay),
                viz_mode=args.viz_mode,
            )
            recording_thread.start()

        phone_thread = PhoneReadThread(phone)
        phone_thread.start()

        _print_banner(args, tactile, agent_overlay_ok, wrist_overlay_ok, frequency)

        last_button_check = 0.0
        button_cooldown = 3.0
        last_button_state = False
        episode_num = 0

        # After a home (script start OR every "Start new episode"), the phone's
        # grasp toggle might still be held ON from the previous episode. Without
        # this latch the gripper would clamp closed on the next env.step. So we
        # force grasp=0 (open) until the user releases the toggle — only THEN
        # do we resume following the toggle's live state.
        _, _initial_grasp, _ = phone_thread.get_data()
        grasp_open_latch = bool(_initial_grasp is not None and _initial_grasp > 0.5)
        if grasp_open_latch:
            print("  Phone toggle is ON. Holding gripper OPEN until you release it.")

        # Live-viz state: poll cached overlays at ~15 Hz, disable on first cv2 error
        # (e.g. no DISPLAY available) so the rest of the session keeps running.
        viz_enabled = bool(args.viz) and recording_thread is not None
        viz_interval = 1.0 / 15.0
        last_viz_time = 0.0

        try:
            while True:
                current_time = time.monotonic()
                if current_time - last_button_check >= button_cooldown:
                    target_pose, grasp_state, button_state = phone_thread.get_data()

                    if args.record and button_state and not last_button_state:
                        if not recording_thread.is_recording():
                            episode_num += 1
                            print(f"  -> Episode {episode_num}: homing...")
                            env.reset(duration=args.reset_duration)
                            for _ in range(5):
                                obs = env.get_obs()
                                time.sleep(0.05)
                            # Re-anchor the phone's AR frame to the freshly homed
                            # robot pose so the operator can hold the phone wherever
                            # and that becomes the new origin.
                            phone.reset(obs["pose"])
                            time.sleep(0.01)  # let PhoneReadThread tick once
                            # CRITICAL: refresh target_pose with the new calibration,
                            # otherwise the env.step at the bottom of this iteration
                            # would command the robot back to wherever the user was
                            # teleoperating before homing.
                            target_pose, grasp_state, _ = phone_thread.get_data()
                            # If the toggle is still held from last episode, latch
                            # the gripper open until the user releases it. Prevents
                            # the gripper from immediately re-closing right after the home.
                            if grasp_state is not None and grasp_state > 0.5:
                                grasp_open_latch = True
                                print("     (toggle still ON; gripper held open until you release it)")
                            recording_thread.set_recording(True)
                            print(f"     recording")
                        else:
                            n_steps = recorder._ep_step_counter
                            recording_thread.set_recording(False)
                            recorder.end_episode()
                            print(f"     done ({n_steps} frames)")
                            print()
                        last_button_check = time.monotonic()

                    last_button_state = button_state
                else:
                    target_pose, grasp_state, _ = phone_thread.get_data()

                # Live viz windows. Pulls the recording_thread's cached native-res
                # overlaid images; if cv2 can't talk to a display (e.g. SSH without
                # X forwarding), disable for the rest of the session instead of
                # crashing the teleop loop.
                if viz_enabled and current_time - last_viz_time >= viz_interval:
                    agent_viz, wrist_viz, _ = recording_thread.get_latest_viz()
                    try:
                        if agent_viz is not None:
                            cv2.imshow("agent (overlay)", agent_viz)
                        if wrist_viz is not None:
                            cv2.imshow("wrist (overlay)", wrist_viz)
                        cv2.waitKey(1)
                    except cv2.error as e:
                        print(f"  [warn] cv2 viz failed (no DISPLAY?): {e}")
                        print(f"  [warn] disabling --viz for the rest of this session")
                        viz_enabled = False
                    last_viz_time = current_time

                if target_pose is None:
                    time.sleep(0.001)
                    continue

                # Open-latch logic: while latched, force grasp=0 (open) regardless of
                # the phone toggle. Latch clears the moment the toggle drops < 0.5.
                if grasp_open_latch:
                    if grasp_state is not None and grasp_state < 0.5:
                        grasp_open_latch = False  # user released the toggle; resume normal
                    else:
                        grasp_state = 0.0          # still holding from before home; keep open

                env.step(target_pose, grasp_state)
                if recording_thread is not None:
                    recording_thread.update_data(target_pose, grasp_state)

                time.sleep(0.01)

        except KeyboardInterrupt:
            print()
            print("Quitting...")
            # Recording thread may exist for --record OR --viz; stop it either way.
            if recording_thread is not None:
                if recorder is not None and recording_thread.is_recording():
                    n_steps = recorder._ep_step_counter
                    recorder.end_episode()
                    print(f"     done ({n_steps} frames)")
                recording_thread.stop()
                recording_thread.join(timeout=2.0)
            if recorder is not None:
                recorder.close()
                print()
                print(f"Saved {episode_num} episode(s) this session.  "
                      f"Dataset now has {recorder.zarr_n} frames in {recorder.path}.")
            phone_thread.stop()
            phone_thread.join(timeout=2.0)
            if args.viz:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
    finally:
        if tactile is not None:
            tactile.__exit__(None, None, None)


if __name__ == "__main__":
    main()
