import time
import numpy as np
import argparse
# from pynput.keyboard import KeyCode  # disabled: needs X display, breaks over SSH. Quit with Ctrl+C instead.

from environment.phone import Phone
from recorder import DatasetRecorder
# from environment.keystroke_counter import KeystrokeCounter  # disabled with pynput above
from tasks.simple_task import Simple_Task
from threads import PhoneReadThread, RecordingThread


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Teleop + recording. Homes the robot to a fixed start pose before each new episode so every demo begins from the same init.')
    parser.add_argument('--record', action='store_true', help='Enable data recording')
    parser.add_argument('--reset-duration', type=float, default=3.0, help='Seconds for the smooth home motion before each episode')
    args = parser.parse_args()

    env = Simple_Task()
    phone = Phone()
    env.reset(duration=args.reset_duration)
    time.sleep(args.reset_duration)

    for _ in range(20):
        obs = env.get_obs()
        time.sleep(0.1)

    phone.reset(obs["pose"])

    frequency = 10.0

    if args.record:
        recorder = DatasetRecorder("teleop_data.zarr", memory_buffer_size=5000, flush_interval=2.0, use_actions=False)
        recording_thread = RecordingThread(recorder, env, frequency)
        recording_thread.start()

    # keystroke_counter = KeystrokeCounter()
    # keystroke_counter.start()

    phone_thread = PhoneReadThread(phone)
    phone_thread.start()

    last_button_check = 0
    button_cooldown = 3.0
    last_button_state = False
    episode_num = 0

    print()
    print("=" * 60)
    print("  READY")
    print("=" * 60)
    if args.record:
        print(f"  Recording   : ON ({frequency:.0f} Hz)  ->  teleop_data.zarr")
    else:
        print(f"  Recording   : OFF  (pass --record to enable)")
    print(f"  Reset pose  : pos=[400, 0, 290]  rot=[180, 0, 0]")
    print(f"  Phone btn A : start / stop episode (robot homes first)")
    print(f"  Ctrl+C      : quit and flush to disk")
    print("=" * 60)
    print()

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
                        # Re-anchor the phone's AR frame to the freshly homed robot pose,
                        # so the operator can hold the phone wherever and that becomes the new origin.
                        phone.reset(obs["pose"])
                        time.sleep(0.01)  # let PhoneReadThread tick once with new calibration
                        # CRITICAL: refresh target_pose with the new calibration, otherwise
                        # the env.step at the bottom of this iteration would command the robot
                        # back to wherever the user was teleoperating before homing.
                        target_pose, grasp_state, _ = phone_thread.get_data()
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

            # q-key quit disabled (pynput needs X). Quit with Ctrl+C — the KeyboardInterrupt handler below does the same cleanup.
            # key_events = keystroke_counter.get_press_events()
            # for key in key_events:
            #     if key == KeyCode(char='q'):
            #         print("Quitting...")
            #         if args.record:
            #             if recording_thread.is_recording():
            #                 recorder.end_episode()
            #             recording_thread.stop()
            #             recording_thread.join(timeout=2.0)
            #             recorder.close()
            #         phone_thread.stop()
            #         phone_thread.join(timeout=2.0)
            #         keystroke_counter.stop()
            #         exit(0)

            if target_pose is None:
                time.sleep(0.001)
                continue

            env.step(target_pose, grasp_state)
            if args.record:
                recording_thread.update_data(target_pose, grasp_state)

            time.sleep(0.01)

    except KeyboardInterrupt:
        print()
        print("Quitting...")
        if args.record:
            if recording_thread.is_recording():
                n_steps = recorder._ep_step_counter
                recorder.end_episode()
                print(f"     done ({n_steps} frames)")
            recording_thread.stop()
            recording_thread.join(timeout=2.0)
            recorder.close()
            print()
            print(f"Saved {episode_num} episode(s) this session.  Dataset now has {recorder.zarr_n} frames in {recorder.path}.")
        phone_thread.stop()
        phone_thread.join(timeout=2.0)
        # keystroke_counter.stop()
