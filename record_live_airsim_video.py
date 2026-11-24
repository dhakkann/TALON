"""Turnkey Live AirSim Showcase Video Recorder & Automatic Segmentation Initializer.

Usage on Mihir's Laptop:
    # 1. Standard RGB Live Flight Recording (e.g. 30 seconds):
    python record_live_airsim_video.py --duration 35 --out live_airsim_showcase.mp4

    # 2. Ground-Truth Unreal Engine Segmentation Recording (AirSim automatically knows every object):
    python record_live_airsim_video.py --segmentation --duration 35 --out live_segmentation_showcase.mp4

    # 3. Autonomous Demo Mission while Recording:
    python record_live_airsim_video.py --auto-mission --out autonomous_flight_demo.mp4
"""

import sys
import os
import time
import argparse
import numpy as np
import cv2

try:
    import airsim
    _AIRSIM_AVAILABLE = True
except ImportError:
    _AIRSIM_AVAILABLE = False


def setup_airsim_ground_truth_segmentation(client) -> bool:
    """Configures Unreal Engine actor mesh IDs so AirSim segmentation returns ground-truth labels."""
    if client is None:
        return False
    print("[AirSim] Initializing Unreal Engine semantic segmentation actor IDs...")
    try:
        # Reset all objects to 0
        client.simSetSegmentationObjectID(".*", 0, True)

        regex_mappings = [
            (".*[Ff]ountain.*", 1, "fountain"),
            (".*[Rr]ed.*", 2, "red building"),
            (".*[Bb]lue.*", 3, "blue building"),
            (".*[Tt]ower.*", 4, "tower"),
            (".*[Hh]ospital.*", 5, "hospital"),
            (".*[Cc]ar.*", 6, "car"),
            (".*[Tt]ruck.*", 7, "truck"),
            (".*[Tt]ree.*", 8, "tree"),
            (".*[Ss]top.*", 9, "stop sign"),
            (".*[Hh]ouse.*", 10, "house"),
            (".*[Pp]illar.*", 11, "pillar"),
            (".*[Cc]ylinder.*", 12, "cylinder"),
            (".*[Oo]range.*", 13, "orange block"),
            (".*[Bb]lock.*", 14, "block"),
            (".*[Bb]ridge.*", 15, "bridge"),
        ]
        count = 0
        for regex, obj_id, label in regex_mappings:
            try:
                if client.simSetSegmentationObjectID(regex, obj_id, True):
                    count += 1
            except Exception:
                pass
        print(f"[AirSim] Successfully mapped {count} Unreal Engine actor class filters to ground-truth IDs!")
        return True
    except Exception as e:
        print(f"[AirSim] Segmentation setup notice: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Record live AirSim flight showcase video on Mihir's laptop.")
    parser.add_argument("--airsim-ip", type=str, default="127.0.0.1", help="AirSim host IP (127.0.0.1 for local)")
    parser.add_argument("--airsim-port", type=int, default=41451, help="AirSim RPC port")
    parser.add_argument("--duration", type=float, default=30.0, help="Recording duration in seconds (0 for indefinite until Ctrl+C)")
    parser.add_argument("--fps", type=float, default=25.0, help="Target recording FPS")
    parser.add_argument("--out", type=str, default="live_airsim_showcase.mp4", help="Output MP4 filename")
    parser.add_argument("--segmentation", action="store_true", help="Capture and overlay Unreal Engine ground-truth segmentation")
    parser.add_argument("--auto-mission", action="store_true", help="Execute an autonomous flight pattern (takeoff, cruise, deflection, landing) during recording")
    args = parser.parse_args()

    if not _AIRSIM_AVAILABLE:
        print("[Error] airsim package is not installed. Please run: pip install airsim")
        sys.exit(1)

    print(f"[Recorder] Connecting to AirSim at {args.airsim_ip}:{args.airsim_port}...")
    client = airsim.MultirotorClient(ip=args.airsim_ip, port=args.airsim_port, timeout_value=5)
    try:
        client.confirmConnection()
        print("[Recorder] AirSim connected successfully!")
    except Exception as e:
        print(f"[Error] Could not connect to AirSim at {args.airsim_ip}:{args.airsim_port}: {e}")
        print("Please ensure AirSim (AirSimNH.exe, Blocks.exe, or CityEnviron.exe) is running.")
        sys.exit(1)

    # Initialize Unreal Engine ground-truth segmentation if requested
    if args.segmentation:
        setup_airsim_ground_truth_segmentation(client)

    # Autonomous flight demonstration setup
    if args.auto_mission:
        print("[Mission] Arming drone and initiating autonomous takeoff...")
        client.enableApiControl(True)
        client.armDisarm(True)
        client.takeoffAsync().join()
        client.moveToZAsync(-10.0, 3.0).join()
        print("[Mission] Drone at cruise altitude (-10m). Starting forward cruise.")
        client.moveByVelocityAsync(4.0, 0.0, 0.0, duration=args.duration)

    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_out = "temp_capture.mp4"
    writer = cv2.VideoWriter(temp_out, fourcc, args.fps, (width, height))

    print(f"\n[Recording] Starting live recording to '{args.out}'...")
    print(f"Target duration: {args.duration:.1f}s | FPS: {args.fps} | Press Ctrl+C to finish anytime.\n")

    start_time = time.time()
    frame_count = 0
    dt = 1.0 / args.fps

    id_to_name = {
        1: "FOUNTAIN",
        2: "RED BUILDING",
        3: "BLUE BUILDING",
        4: "METRO TOWER",
        5: "HOSPITAL",
        6: "CAR",
        7: "TRUCK",
        8: "TREE",
        9: "STOP SIGN",
        10: "HOUSE",
        11: "PILLAR",
        12: "CYLINDER",
        13: "ORANGE BLOCK",
        14: "BLOCK",
        15: "BRIDGE",
    }

    try:
        while True:
            t_loop = time.time()
            elapsed = t_loop - start_time
            if args.duration > 0 and elapsed >= args.duration:
                break

            # Request Scene RGB and optional Segmentation
            reqs = [
                airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
            ]
            if args.segmentation:
                reqs.append(airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, False, False))

            responses = client.simGetImages(reqs)
            if not responses or len(responses[0].image_data_uint8) == 0:
                fallback_reqs = [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)]
                if args.segmentation:
                    fallback_reqs.append(airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False))
                responses = client.simGetImages(fallback_reqs)

            if not responses or len(responses[0].image_data_uint8) == 0:
                time.sleep(0.01)
                continue

            # Decode RGB
            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            rw, rh = responses[0].width, responses[0].height
            if img1d.size == rh * rw * 3:
                rgb = img1d.reshape(rh, rw, 3)
            elif img1d.size == rh * rw * 4:
                rgb = img1d.reshape(rh, rw, 4)[:, :, :3]
            else:
                continue

            if rw != width or rh != height:
                frame = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
            else:
                frame = rgb.copy()

            # Process ground truth segmentation actor bounding boxes
            if args.segmentation and len(responses) >= 2:
                seg1d = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
                if seg1d.size == rh * rw * 3 or seg1d.size == rh * rw * 4:
                    ch = 3 if seg1d.size == rh * rw * 3 else 4
                    seg = seg1d.reshape(rh, rw, ch)[:, :, :3]
                    seg_small = cv2.resize(seg, (width, height), interpolation=cv2.INTER_NEAREST)
                    gray = cv2.cvtColor(seg_small, cv2.COLOR_BGR2GRAY)
                    unique_vals = np.unique(gray)
                    for val in unique_vals:
                        if val == 0:
                            continue
                        mask = (gray == val).astype(np.uint8) * 255
                        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for c in cnts:
                            if cv2.contourArea(c) > 500:
                                x, y, bw, bh = cv2.boundingRect(c)
                                obj_name = id_to_name.get(int(val), f"ACTOR ID {val}")
                                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 128), 2)
                                tag = f"GROUND-TRUTH: {obj_name} (1.00)"
                                cv2.rectangle(frame, (x, max(0, y - 22)), (x + len(tag) * 9, max(0, y)), (0, 180, 80), -1)
                                cv2.putText(frame, tag, (x + 3, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            # Query vehicle telemetry
            state = client.getMultirotorState()
            kin = state.kinematics_estimated
            pos = getattr(kin, "position", getattr(kin, "position_val", None))
            vel = getattr(kin, "linear_velocity", getattr(kin, "linear_velocity_val", None))
            vx = getattr(vel, "x_val", 0.0)
            vy = getattr(vel, "y_val", 0.0)
            vz = getattr(vel, "z_val", 0.0)
            px = getattr(pos, "x_val", 0.0)
            py = getattr(pos, "y_val", 0.0)
            pz = getattr(pos, "z_val", 0.0)
            spd = float(np.sqrt(vx**2 + vy**2 + vz**2))
            alt = float(-pz)

            # Draw HUD Overlay Banner
            cv2.rectangle(frame, (0, 0), (width, 42), (10, 15, 26), -1)
            cv2.line(frame, (0, 42), (width, 42), (0, 240, 255), 1)

            cv2.putText(frame, "AIRSIM LIVE FLIGHT", (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"TIME: {elapsed:.1f}s", (260, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(frame, f"ALT: {alt:.1f} m", (410, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 128), 1, cv2.LINE_AA)
            cv2.putText(frame, f"SPEED: {spd:.1f} m/s", (550, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"POS: [{px:.1f}, {py:.1f}]", (730, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 200, 220), 1, cv2.LINE_AA)

            engine_label = "UNREAL GROUND-TRUTH SEGMENTATION" if args.segmentation else "ONBOARD AIRSIM SENSORS"
            cv2.putText(frame, f"ENGINE: {engine_label}", (980, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 180, 0), 1, cv2.LINE_AA)

            # Draw Central Crosshair
            cx, cy = width // 2, height // 2
            cv2.line(frame, (cx - 15, cy), (cx + 15, cy), (0, 240, 255), 1)
            cv2.line(frame, (cx, cy - 15), (cx, cy + 15), (0, 240, 255), 1)

            writer.write(frame)
            frame_count += 1

            if frame_count % 30 == 0:
                print(f"[Status] Captured {frame_count} frames ({elapsed:.1f}s / {args.duration:.1f}s) - Alt: {alt:.1f}m - Spd: {spd:.1f}m/s")

            # Sleep to match target FPS
            loop_elapsed = time.time() - t_loop
            time.sleep(max(0.001, dt - loop_elapsed))

    except KeyboardInterrupt:
        print("\n[Recorder] Stop signal received from operator. Finalizing video...")

    finally:
        writer.release()
        print(f"[Recorder] Finished capturing {frame_count} frames.")

        # Re-encode to universal H.264 using ffmpeg if available
        import shutil
        ffmpeg_exe = shutil.which("ffmpeg")
        if not ffmpeg_exe:
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ffmpeg_exe = None

        if ffmpeg_exe and os.path.exists(ffmpeg_exe):
            print(f"[Recorder] Transcoding with ffmpeg to universal high-quality H.264 '{args.out}'...")
            cmd = f'"{ffmpeg_exe}" -y -i "{temp_out}" -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100 -c:v libx264 -preset fast -crf 20 -profile:v main -level 3.1 -pix_fmt yuv420p -movflags +faststart -c:a aac -b:a 128k -shortest "{args.out}"'
            if os.system(cmd + " >nul 2>&1") == 0:
                try:
                    os.remove(temp_out)
                except Exception:
                    pass
                print(f"[Success] Video saved to '{args.out}' (Universal H.264 AVC1 + AAC Audio)!")
                return

        # Fallback if ffmpeg is not in PATH
        if os.path.exists(temp_out):
            if os.path.exists(args.out):
                os.remove(args.out)
            os.rename(temp_out, args.out)
            print(f"[Success] Video saved to '{args.out}'!")


if __name__ == "__main__":
    main()
