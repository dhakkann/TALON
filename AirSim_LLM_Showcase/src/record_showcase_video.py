"""Authentic Live AirSimInspection.exe Showcase Video Recorder (24 FPS Stepped Capture).

Captures smooth 24 FPS live Unreal Engine FPV RGB + DepthPlanar telemetry frames
across every stage of a language-guided autonomous inspection mission using
deterministic AirSim simulation stepping (`simPause` + `simContinueForTime`) so
even integrated GPUs (e.g., Intel Iris Xe) produce a stutter-free H.264 MP4.
"""

import argparse
import subprocess
import time
from pathlib import Path

import airsim
import cv2
import numpy as np

try:
    from airsim_wrapper import AirSimWrapper, colorize_depth, compute_front_distance
    from gemini_airsim import DroneChatAgent
except ImportError:
    from src.airsim_wrapper import AirSimWrapper, colorize_depth, compute_front_distance
    from src.gemini_airsim import DroneChatAgent


def draw_hud(
    rgb_bgr: np.ndarray,
    depth_bgr: np.ndarray,
    pos: list[float],
    yaw: float,
    front_dist: float,
    stage_title: str,
    user_prompt: str,
    bot_reply: str,
    avoiding: bool,
) -> np.ndarray:
    """Compose a 1280x720 split-panel live telemetry + RGB + DepthPlanar frame."""
    canvas = np.full((720, 1280, 3), (14, 20, 30), dtype=np.uint8)

    # Top header bar (clear spacing so title never overlaps status badge)
    cv2.rectangle(canvas, (0, 0), (1280, 56), (20, 30, 44), -1)
    cv2.putText(
        canvas,
        "AR511 Language-Guided Autonomous Drone | AirSimInspection Live",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (230, 240, 252),
        2,
    )
    badge_color = (60, 70, 235) if avoiding else (60, 195, 95)
    badge_text = "AVOIDING OBSTACLE" if avoiding else stage_title
    (tw, _), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    badge_x = max(830, 1260 - tw)
    cv2.putText(
        canvas,
        badge_text,
        (badge_x, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        badge_color,
        2,
    )

    # Left: Live Unreal RGB FPV (620x465)
    rgb_resized = cv2.resize(rgb_bgr, (620, 465), interpolation=cv2.INTER_CUBIC)
    canvas[72:537, 20:640] = rgb_resized
    cv2.rectangle(canvas, (20, 72), (640, 537), (45, 65, 90), 2)
    cv2.putText(
        canvas,
        "LIVE FPV RGB (AirSimInspection.exe)",
        (30, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    # Right: Live DepthPlanar Heatmap (600x390) + Telemetry
    depth_resized = cv2.resize(depth_bgr, (600, 390), interpolation=cv2.INTER_LINEAR)
    canvas[72:462, 660:1260] = depth_resized
    cv2.rectangle(canvas, (660, 72), (1260, 462), (45, 65, 90), 2)
    cv2.putText(
        canvas,
        f"LIVE DepthPlanar | Front Clearance: {front_dist:.1f} m",
        (672, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    # Telemetry box under depth
    cv2.rectangle(canvas, (660, 474), (1260, 537), (22, 32, 46), -1)
    cv2.rectangle(canvas, (660, 474), (1260, 537), (45, 65, 90), 1)
    tel_line = (
        f"POS: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) m  |  "
        f"YAW: {yaw:.1f} deg  |  DIST: {front_dist:.1f} m"
    )
    cv2.putText(
        canvas,
        tel_line,
        (672, 512),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (140, 215, 255),
        1,
    )

    # Bottom dialogue & generated code strip
    cv2.rectangle(canvas, (20, 550), (1260, 702), (18, 26, 38), -1)
    cv2.rectangle(canvas, (20, 550), (1260, 702), (45, 65, 90), 1)
    cv2.putText(
        canvas,
        f"USER > {user_prompt[:95]}",
        (34, 584),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (110, 225, 255),
        2,
    )
    for i, line in enumerate(bot_reply.splitlines()[:3]):
        cv2.putText(
            canvas,
            line[:115],
            (34, 622 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (225, 235, 245) if "Question" not in line else (100, 200, 255),
            1,
        )
    return canvas


def capture_hud_frame(
    aw: AirSimWrapper,
    stage_title: str,
    user_prompt: str,
    bot_reply: str,
    avoiding: bool = False,
) -> np.ndarray:
    """Grab live RGB + DepthPlanar from AirSimInspection and compose HUD frame."""
    responses = aw.client.simGetImages(
        [
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False),
        ]
    )
    r0, r1 = responses[0], responses[1]
    rgb = np.frombuffer(r0.image_data_uint8, dtype=np.uint8).reshape(
        r0.height, r0.width, 3
    )
    depth_arr = np.array(r1.image_data_float, dtype=np.float32).reshape(
        r1.height, r1.width
    )
    front_dist = compute_front_distance(depth_arr)
    depth_bgr = colorize_depth(depth_arr)
    pos = aw.get_drone_position()
    yaw = aw.get_yaw()
    return draw_hud(
        rgb, depth_bgr, pos, yaw, front_dist, stage_title, user_prompt, bot_reply, avoiding
    )


class SteppedMissionRecorder:
    """Writes one smooth HUD frame per simulation step at target FPS."""

    def __init__(self, aw: AirSimWrapper, writer: cv2.VideoWriter, fps: float):
        self.aw = aw
        self.writer = writer
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.frame_count = 0
        self.stage_title = "INIT"
        self.user_prompt = ""
        self.bot_reply = ""
        self.avoiding = False

    def set_hud(
        self,
        stage_title: str,
        user_prompt: str,
        bot_reply: str,
        avoiding: bool = False,
    ):
        self.stage_title = stage_title
        self.user_prompt = user_prompt
        self.bot_reply = bot_reply
        self.avoiding = avoiding

    def snap(self):
        frame = capture_hud_frame(
            self.aw,
            self.stage_title,
            self.user_prompt,
            self.bot_reply,
            avoiding=self.avoiding,
        )
        self.writer.write(frame)
        self.frame_count += 1

    def record_hover(self, seconds: float):
        steps = max(1, int(round(seconds * self.fps)))
        for _ in range(steps):
            self.snap()
            self.aw.client.simContinueForTime(self.dt)


def main():
    parser = argparse.ArgumentParser(
        description="Record smooth 24 FPS live AirSimInspection showcase MP4"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent
            / "demo_recordings"
            / "airsim_llm_showcase_demo.mp4"
        ),
    )
    parser.add_argument("--fps", type=float, default=24.0)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_mp4 = out_path.with_name("raw_capture.mp4")

    aw = AirSimWrapper()
    aw.client.simPause(False)
    aw.client.reset()
    time.sleep(1.0)
    aw.client.enableApiControl(True)
    aw.client.armDisarm(True)

    agent = DroneChatAgent(offline=False)
    writer = cv2.VideoWriter(
        str(raw_mp4), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1280, 720)
    )
    rec = SteppedMissionRecorder(aw, writer, args.fps)

    try:
        # Pause clock so every frame advances cleanly by exact 1/fps simulation seconds
        aw.client.simPause(True)
        aw.set_step_recorder(rec.snap, dt=rec.dt)

        # Stage 1: Takeoff & climb to 12 meters
        print("[Rec] Stage 1: Takeoff & climb...", flush=True)
        p1 = "Take off and climb to 12 meters"
        r1 = agent.ask(p1)
        print(f"[Rec] Stage 1 ({r1['source']}): {r1['code']}", flush=True)
        rec.set_hud("STAGE 1: TAKEOFF", p1, r1["code"] or r1["text"])
        aw.takeoff()
        rec.set_hud("STAGE 1: CLIMB 12M", p1, r1["code"] or r1["text"])
        aw.fly_to([0.0, 0.0, 12.0], speed=8.0)
        rec.record_hover(0.8)

        # Stage 2: Interactive Clarification (turbines spin smoothly in background!)
        print("[Rec] Stage 2: Clarification...", flush=True)
        p2 = "Fly to the turbine"
        r2 = agent.ask(p2)
        print(f"[Rec] Stage 2 ({r2['source']}): {r2['text']}", flush=True)
        rec.set_hud("STAGE 2: CLARIFY", p2, r2["text"])
        rec.record_hover(1.6)

        # Reposition swiftly toward turbine1 (paused capture during long open-field transit)
        t1 = aw.get_position("turbine1")
        aw.set_step_recorder(None)
        aw.client.simPause(False)
        aw.fly_to([t1[0] - 24.0, t1[1], 12.0], speed=14.0)
        aw.set_yaw(0.0)
        aw.client.simPause(True)
        aw.set_step_recorder(rec.snap, dt=rec.dt)

        # Stage 3: Live fly_safe() DepthPlanar Obstacle Avoidance around turbine1
        print("[Rec] Stage 3: Live fly_safe() around turbine1...", flush=True)
        p3 = "Turbine 1 please, fly past it safely at 12m altitude"
        r3 = agent.ask(p3)
        base_code = r3["code"] or r3["text"]
        min_clr = 7.0
        rec.set_hud("STAGE 3: TURBINE 1", p3, base_code, avoiding=False)

        def on_step(avoiding: bool, clearance: float):
            if avoiding:
                lookahead = min_clr + 3.0
                msg = (
                    f"DepthPlanar clearance={clearance:.1f}m < {lookahead:.1f}m "
                    f"(min={min_clr:.1f}m) -> sidestepping right (+6m) & climbing (+3m)"
                )
                rec.set_hud("OBSTACLE AVOIDANCE", p3, msg, avoiding=True)
            else:
                rec.set_hud("STAGE 3: TURBINE 1", p3, base_code, avoiding=False)

        aw.fly_safe(
            [t1[0] + 15.0, t1[1], 12.0],
            step_size=10.0,
            min_clearance=min_clr,
            max_steps=30,
            speed=10.0,
            on_step=on_step,
        )
        rec.set_hud("STAGE 3: CLEAR", p3, base_code, avoiding=False)
        rec.record_hover(0.8)

        # Stage 4: Approach & Inspect Car from 8m above
        print("[Rec] Stage 4: Car inspection...", flush=True)
        p4 = "Inspect the car from 8 meters above"
        r4 = agent.ask(p4)
        car = aw.get_position("car")
        car_z = max(car[2] + 8.0, 8.0)
        # Move quickly into car approach zone so final approach & overhead inspection stay crisp
        aw.set_step_recorder(None)
        aw.client.simPause(False)
        aw.fly_to([car[0] + 18.0, car[1] - 18.0, car_z + 2.0], speed=14.0)
        aw.client.simPause(True)
        aw.set_step_recorder(rec.snap, dt=rec.dt)
        rec.set_hud("STAGE 4: INSPECT CAR", p4, r4["code"] or r4["text"], avoiding=False)
        aw.fly_safe(
            [car[0], car[1], car_z],
            step_size=12.0,
            min_clearance=7.0,
            max_steps=15,
            speed=9.0,
        )
        rec.record_hover(0.8)

        # Stage 5: Smooth descent & full touchdown landing
        print("[Rec] Stage 5: Landing...", flush=True)
        p5 = "Land the drone safely"
        r5 = agent.ask(p5)
        rec.set_hud("STAGE 5: LANDING", p5, r5["code"] or r5["text"], avoiding=False)
        aw.land()
        rec.set_hud("STAGE 5: LANDED", p5, r5["code"] or r5["text"], avoiding=False)
        rec.record_hover(1.2)
    finally:
        aw.set_step_recorder(None)
        try:
            aw.client.simPause(False)
        except Exception:
            pass
        writer.release()

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(raw_mp4),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0 and out_path.exists():
        raw_mp4.unlink(missing_ok=True)
        print(
            f"Saved {args.fps:.0f}fps H.264 MP4 ({rec.frame_count} frames): "
            f"{out_path} ({out_path.stat().st_size} bytes)"
        )
    else:
        raw_mp4.replace(out_path)
        print(f"Saved MP4 (fallback): {out_path} | ffmpeg stderr: {proc.stderr[:200]}")


if __name__ == "__main__":
    main()
