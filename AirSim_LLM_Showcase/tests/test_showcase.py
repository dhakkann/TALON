import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from airsim_wrapper import AirSimWrapper, colorize_depth, compute_front_distance  # noqa: E402
from gemini_airsim import (  # noqa: E402
    DroneChatAgent,
    OfflineDronePlanner,
    extract_python_code,
    resolve_gemini_api_key,
    write_chat_history,
)
from web_cockpit import create_app  # noqa: E402


def test_compute_front_distance_edge_cases():
    assert compute_front_distance(None) == 100.0
    assert compute_front_distance(np.array([], dtype=np.float32)) == 100.0
    all_nan = np.full((120, 160), np.nan, dtype=np.float32)
    assert compute_front_distance(all_nan) == 100.0

    # Ground pixels at very bottom should not trigger false obstacle inside central ROI
    clear = np.full((120, 160), 50.0, dtype=np.float32)
    clear[100:, :] = 0.8
    assert compute_front_distance(clear) == pytest.approx(50.0, abs=0.1)

    # Obstacle in central corridor triggers close reading
    blocked = np.full((120, 160), 60.0, dtype=np.float32)
    blocked[40:75, 60:100] = 2.4
    assert compute_front_distance(blocked) < 3.0


def test_colorize_depth_shape_and_handles_nan():
    arr = np.array([[np.nan, 5.0], [np.inf, 20.0]], dtype=np.float32)
    img = colorize_depth(arr)
    assert img.shape == (2, 2, 3)
    assert img.dtype == np.uint8


def test_fly_safe_sidesteps_and_reaches_goal():
    aw = AirSimWrapper(connect=False)
    pos = [0.0, 0.0, 10.0]
    yaw = [0.0]
    visited = []
    distances = [2.0, 25.0, 25.0, 25.0]  # Blocked on first check, clear after sidestep

    def fake_get_pos():
        return list(pos)

    def fake_fly_to(pt, speed=5.0):
        pos[0], pos[1], pos[2] = float(pt[0]), float(pt[1]), float(pt[2])
        visited.append(list(pos))

    def fake_get_yaw():
        return yaw[0]

    def fake_set_yaw(val):
        yaw[0] = float(val)

    def fake_get_distance():
        return distances.pop(0) if distances else 25.0

    aw.get_drone_position = fake_get_pos
    aw.fly_to = fake_fly_to
    aw.get_yaw = fake_get_yaw
    aw.set_yaw = fake_set_yaw
    aw.get_distance = fake_get_distance

    reached = aw.fly_safe([12.0, 0.0, 10.0], step_size=6.0, min_clearance=4.5, max_steps=10)
    assert reached is True
    assert len(visited) >= 2
    # First move should climb +3m and sidestep +6m in Y (since yaw=0 -> right is +Y)
    assert visited[0][2] == pytest.approx(13.0)
    assert visited[0][1] == pytest.approx(6.0)
    assert visited[-1][:2] == pytest.approx([12.0, 0.0], abs=0.1)


def test_fly_safe_returns_false_when_blocked():
    aw = AirSimWrapper(connect=False)
    pos = [0.0, 0.0, 10.0]
    aw.get_drone_position = lambda: list(pos)
    aw.fly_to = lambda pt, speed=5.0: pos.__setitem__(slice(None), [pt[0], pt[1], pt[2]])
    aw.get_yaw = lambda: 0.0
    aw.set_yaw = lambda val: None
    aw.get_distance = lambda: 1.5  # Always blocked

    assert aw.fly_safe([50.0, 0.0, 10.0], max_steps=3) is False


def test_fly_safe_caps_step_to_avoid_jumping_into_obstacle():
    """Obstacle at x=112.5 (6.5m ahead) must NOT be jumped over by a 10m step even if min_clearance=4.5."""
    aw = AirSimWrapper(connect=False)
    pos = [106.0, -43.0, 12.0]
    visited = []

    aw.get_drone_position = lambda: list(pos)
    aw.get_yaw = lambda: 0.0
    aw.set_yaw = lambda val: None

    def fake_get_distance():
        # Obstacle pillar at x in [112.5, 117.5], y < -40.0
        if pos[1] < -40.0 and pos[0] < 112.5:
            return 112.5 - pos[0]
        return 40.0

    def fake_fly_to(pt, speed=9.0):
        pos[0], pos[1], pos[2] = float(pt[0]), float(pt[1]), float(pt[2])
        visited.append(list(pos))

    aw.get_distance = fake_get_distance
    aw.fly_to = fake_fly_to
    reached = aw.fly_safe([130.0, -43.0, 12.0], step_size=10.0, min_clearance=4.5, max_steps=10)
    assert reached is True
    # Drone must never enter the obstacle footprint [112.5..117.5, y < -40.0]
    assert all(not (p[1] < -40.0 and 112.5 <= p[0] <= 117.5) for p in visited)
    assert any(p[1] > -43.0 and p[2] > 12.0 for p in visited)


def test_fly_safe_respects_abort():
    aw = AirSimWrapper(connect=False)
    aw.get_drone_position = lambda: [0.0, 0.0, 10.0]
    aw.request_abort()
    assert aw.fly_safe([50.0, 0.0, 10.0]) is False


def test_takeoff_and_land_altitude_handling():
    aw = AirSimWrapper(connect=False)
    aw.client = MagicMock()
    aw.refresh_snapshot = lambda: None
    aw._refresh_snapshot_unlocked = lambda: None

    # Takeoff: if takeoffAsync leaves drone low (-1.0m), takeoff climbs to -2.5 NED
    aw.get_drone_position = lambda: [0.0, 0.0, -1.0]
    aw.takeoff()
    aw.client.takeoffAsync.assert_called_once()
    aw.client.moveToPositionAsync.assert_called_once()

    # Land from 10m: must first descend via moveToPositionAsync before landAsync
    aw.client.reset_mock()
    aw.get_drone_position = lambda: [0.0, 0.0, 10.0]
    aw.land()
    aw.client.moveToPositionAsync.assert_called_once()
    aw.client.landAsync.assert_called_once()


def test_wait_future_stepped_recorder_steps_simulation():
    aw = AirSimWrapper(connect=False)
    aw.client = MagicMock()
    snaps = []

    class FakeFuture:
        def __init__(self):
            self._set_flag = False
            self.calls = 0

        def join(self):
            self._set_flag = True

    fut = FakeFuture()

    def continue_side_effect(dt):
        fut.calls += 1
        if fut.calls >= 3:
            fut._set_flag = True

    aw.client.simContinueForTime.side_effect = continue_side_effect
    aw.set_step_recorder(lambda: snaps.append(1), dt=1.0 / 24.0)
    aw._wait_future(fut, max_wait_sec=1.0)
    assert len(snaps) == 3
    assert aw.client.simContinueForTime.call_count == 3


def test_compute_front_distance_80x60_depth_shape():
    depth_80x60 = np.full((60, 80), 45.0, dtype=np.float32)
    assert compute_front_distance(depth_80x60) == pytest.approx(45.0, abs=0.1)
    depth_80x60[22:38, 30:50] = 6.2
    assert compute_front_distance(depth_80x60) < 7.0


def test_get_position_rejects_unknown_and_missing_actor():
    aw = AirSimWrapper(connect=False)
    aw.client = MagicMock()
    aw.client.simListSceneObjects.return_value = []
    with pytest.raises(ValueError):
        aw.get_position("nonexistent_object")
    with pytest.raises(RuntimeError):
        aw.get_position("solar panel")  # alias resolves to solarpanels, bounded retry fails cleanly


def test_extract_python_code_variants():
    assert extract_python_code("no code here") is None
    assert extract_python_code("```python\naw.takeoff()\n```") == "aw.takeoff()"
    assert extract_python_code("```py\nx = 1\n```\ntext") == "x = 1"
    assert extract_python_code("```\naw.land()\n```") == "aw.land()"


def test_write_chat_history_persists_cleanly(tmp_path):
    history = [
        {"role": "system", "text": "sys"},
        {"role": "user", "text": "take off"},
        {"role": "assistant", "text": "```python\naw.takeoff()\n```"},
    ]
    out = write_chat_history("gemini-2.5-flash", "unit_test", history, chats_dir=tmp_path)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "system:" in content
    assert "aw.takeoff()" in content


def test_resolve_gemini_api_key_env_and_placeholder(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    cfg.write_text('{"GEMINI_API_KEY": "API-KEY"}', encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "test-env-key-123")
    assert resolve_gemini_api_key(cfg) == "test-env-key-123"


def test_offline_planner_clarification_and_targets():
    planner = OfflineDronePlanner()

    r1 = planner.respond("Fly to the turbine")
    assert r1.startswith("Question -")
    assert extract_python_code(r1) is None

    r2 = planner.respond("1")
    code2 = extract_python_code(r2)
    assert code2 is not None
    assert 'aw.get_position("turbine1")' in code2
    assert "aw.fly_safe" in code2

    r3 = planner.respond("Inspect the tower please")
    assert r3.startswith("Question -")
    r4 = planner.respond("third one")
    assert 'aw.get_position("tower3")' in extract_python_code(r4)

    r5 = planner.respond("Fly a 15m circle orbit")
    assert "aw.fly_path" in extract_python_code(r5)

    # Edge cases: "carefully" must NOT match "car"; "10m" must NOT resolve turbine ambiguity
    r6 = planner.respond("Fly 10m toward the turbine")
    assert r6.startswith("Question -")
    r7 = planner.respond("Fly carefully to the solar panels")
    assert 'aw.get_position("solarpanels")' in extract_python_code(r7)
    r8 = planner.respond("Take off and hover at 8 meters")
    assert "8.0" in extract_python_code(r8)


def test_agent_falls_back_on_gemini_error():
    fake_client = MagicMock()
    fake_chat = MagicMock()
    fake_chat.send_message.side_effect = RuntimeError("Network down")
    fake_client.chats.create.return_value = fake_chat

    agent = DroneChatAgent(offline=False, genai_client=fake_client)
    res = agent.ask("Fly to the car")
    assert res["source"] == "offline_fallback"
    assert 'aw.get_position("car")' in res["code"]


def test_web_cockpit_dry_run_endpoints(tmp_path, monkeypatch):
    import web_cockpit

    monkeypatch.setattr(
        web_cockpit,
        "write_chat_history",
        lambda m, t, h: write_chat_history(m, t, h, chats_dir=tmp_path),
    )
    app = create_app(offline=True, dry_run=True, testname="unit_web")
    client = app.test_client()

    assert client.get("/").status_code == 200
    assert client.post("/api/chat", json={"message": ""}).status_code == 400

    q_res = client.post("/api/chat", json={"message": "Fly to the turbine"}).get_json()
    assert q_res["is_question"] is True
    assert q_res["code"] is None

    a_res = client.post("/api/chat", json={"message": "turbine 2"}).get_json()
    assert a_res["is_question"] is False
    assert 'aw.get_position("turbine2")' in a_res["code"]

    tel = client.get("/api/telemetry").get_json()
    assert tel["dry_run"] is True
    assert "front_distance" in tel

    assert client.get("/api/frame/rgb.jpg").status_code == 200
    assert client.get("/api/frame/depth.jpg").status_code == 200
    assert client.post("/api/land").status_code == 200
    save_res = client.post("/api/save").get_json()
    assert Path(save_res["path"]).exists()
