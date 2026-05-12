import time
import numpy as np
import argparse
from pynput.keyboard import KeyCode

from environment.phone import Phone
from recorder import DatasetRecorder
from environment.keystroke_counter import KeystrokeCounter
from tasks.pushT_task import PushT_Task
from tasks.simple_task import Simple_Task
from threads import PhoneReadThread, RecordingThread


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Control robot with optional data recording')
    parser.add_argument('--record', action='store_true', help='Enable data recording')
    args = parser.parse_args()
    
    env = Simple_Task()
    phone = Phone()
    env.reset(duration=3.0)
    time.sleep(3.0)

    for _ in range(50):
        obs = env.get_obs()
        time.sleep(0.1)
    
    phone.reset(obs["pose"])
    
    frequency = 10.0
    
    if args.record:
        recorder = DatasetRecorder("teleop_data.zarr", memory_buffer_size=5000, flush_interval=2.0, use_actions=False)
        recording_thread = RecordingThread(recorder, env, frequency)
        recording_thread.start()
    
    keystroke_counter = KeystrokeCounter()
    keystroke_counter.start()
    
    phone_thread = PhoneReadThread(phone)
    phone_thread.start()
    
    last_button_check = 0
    button_cooldown = 3.0
    last_button_state = False
    
    print("Phone Button - Start/Stop recording episode" if args.record else "Phone Button - Disabled (no recording)")
    print(f"Recording frequency: {frequency} Hz" if args.record else "Recording: Disabled")
    print("[Q] - Quit")
    
    try:
        while True:
            current_time = time.monotonic()
            if current_time - last_button_check >= button_cooldown:
                target_pose, grasp_state, button_state = phone_thread.get_data()
                
                if args.record and button_state and not last_button_state:
                    if not recording_thread.is_recording():
                        recording_thread.set_recording(True)
                        print("Started recording episode")
                    else:
                        recording_thread.set_recording(False)
                        recorder.end_episode()
                        # env.reset(duration=3.0)
                        print("Stopped recording episode")
                    last_button_check = current_time
                
                last_button_state = button_state
            else:
                target_pose, grasp_state, _ = phone_thread.get_data()
            
            key_events = keystroke_counter.get_press_events()
            for key in key_events:
                if key == KeyCode(char='q'):
                    print("Quitting...")
                    if args.record:
                        if recording_thread.is_recording():
                            recorder.end_episode()
                        recording_thread.stop()
                        recording_thread.join(timeout=2.0)
                        recorder.close()
                    phone_thread.stop()
                    phone_thread.join(timeout=2.0)
                    keystroke_counter.stop()
                    exit(0)
            
            if target_pose is None:
                time.sleep(0.001)
                continue

            # target = target_pose[:3]
            env.step(target_pose, grasp_state)
            if args.record:
                recording_thread.update_data(target_pose, grasp_state)
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("Interrupted! Saving data...")
        if recording_thread.is_recording():
            recorder.end_episode()
        recording_thread.stop()
        recording_thread.join(timeout=2.0)
        phone_thread.stop()
        phone_thread.join(timeout=2.0)
        recorder.close()
        keystroke_counter.stop()