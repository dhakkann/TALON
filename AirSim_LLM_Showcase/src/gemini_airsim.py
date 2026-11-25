import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np

try:
    from google import genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except ImportError:
    genai = None
    types = None
    _GENAI_AVAILABLE = False

try:
    from airsim_wrapper import AirSimWrapper
except ImportError:
    from src.airsim_wrapper import AirSimWrapper


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT_PATH = REPO_ROOT / "prompts" / "airsim_basic.txt"
DEFAULT_SYSPROMPT_PATH = REPO_ROOT / "system_prompts" / "airsim_basic.txt"
DEFAULT_CONFIG_PATH = SRC_DIR / "config.json"
DEFAULT_CHATS_DIR = REPO_ROOT / "chats" / "gemini"
DEFAULT_MODEL = "gemini-3.6-flash"
FALLBACK_GEMINI_MODELS = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite")


class colors:
    RED = "\033[31m"
    ENDC = "\033[m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"


def resolve_gemini_api_key(config_path: Path = DEFAULT_CONFIG_PATH) -> str | None:
    """Resolve GEMINI_API_KEY from env, Windows user/system registry, or non-placeholder config."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key and key != "API-KEY":
        return key

    if os.name == "nt":
        try:
            import winreg

            for root, subkey in (
                (winreg.HKEY_CURRENT_USER, "Environment"),
                (
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
                ),
            ):
                try:
                    with winreg.OpenKey(root, subkey) as reg_key:
                        val, _ = winreg.QueryValueEx(reg_key, "GEMINI_API_KEY")
                        if isinstance(val, str) and val.strip() and val.strip() != "API-KEY":
                            return val.strip()
                except OSError:
                    continue
        except Exception:
            pass

    try:
        if Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg_key = str(cfg.get("GEMINI_API_KEY", "")).strip()
            if cfg_key and cfg_key != "API-KEY":
                return cfg_key
    except Exception:
        pass

    return None


def extract_python_code(content: str) -> str | None:
    """Extract Python code from markdown fenced blocks."""
    if not content:
        return None
    matches = re.findall(r"```(?:python|py)?\s*\r?\n?(.*?)```", content, re.DOTALL | re.IGNORECASE)
    if not matches:
        return None
    full_code = "\n".join(m.strip("\r\n") for m in matches if m.strip())
    return full_code if full_code else None


def is_clarification_question(content: str) -> bool:
    if not content:
        return False
    stripped = content.strip()
    return stripped.lower().startswith("question") or "\nquestion -" in stripped.lower()


def write_chat_history(
    model_name: str,
    test_name: str,
    chat_history: list[dict],
    chats_dir: Path = DEFAULT_CHATS_DIR,
) -> Path:
    """Persist conversation history safely to chats/gemini/."""
    chats_dir = Path(chats_dir)
    chats_dir.mkdir(parents=True, exist_ok=True)
    safe_model = re.sub(r"[^a-zA-Z0-9._-]", "_", model_name)
    safe_test = re.sub(r"[^a-zA-Z0-9._-]", "_", test_name)
    path = chats_dir / f"{safe_model}-{safe_test}.txt"
    with open(path, "w", encoding="utf-8") as f:
        for entry in chat_history:
            role = entry.get("role", "unknown")
            text = entry.get("text", "")
            f.write(f"{role}: \n{text}\n")
            f.write("\n<------>\n\n")
    return path


class OfflineDronePlanner:
    """Deterministic offline preset command & clarification fallback."""

    def __init__(self):
        self.pending_ambiguous: str | None = None

    @staticmethod
    def _extract_number(text: str, default: float = 10.0) -> float:
        m = re.search(r"(-?\d+(?:\.\d+)?)", text)
        return float(m.group(1)) if m else default

    @staticmethod
    def _target_code(target: str) -> tuple[str, str]:
        if target in ("solarpanels", "car", "crowd"):
            code = (
                f'pos = aw.get_position("{target}")\n'
                f"aw.fly_safe([pos[0], pos[1], max(pos[2] + 10.0, 8.0)])"
            )
            desc = f"Safely flies above `{target}` (10m vertical clearance) using `aw.fly_safe()`."
        else:
            code = (
                f'pos = aw.get_position("{target}")\n'
                f"aw.fly_safe([pos[0] - 15.0, pos[1], max(pos[2] + 15.0, 12.0)])"
            )
            desc = (
                f"Safely approaches `{target}` at 15m X-offset and 15m altitude using "
                f"`aw.fly_safe()` with DepthPlanar obstacle avoidance."
            )
        return code, desc

    def _format(self, code: str, explanation: str) -> str:
        return f"```python\n{code}\n```\n\n[Offline Preset] {explanation}"

    def respond(self, user_text: str) -> str:
        text = user_text.strip().lower()

        # Resolve pending clarification follow-up
        if self.pending_ambiguous == "turbine":
            if re.search(r"\b(2|second|two|turbine\s*2)\b", text):
                self.pending_ambiguous = None
                code, desc = self._target_code("turbine2")
                return self._format(code, desc)
            if re.search(r"\b(1|first|one|turbine\s*1)\b", text):
                self.pending_ambiguous = None
                code, desc = self._target_code("turbine1")
                return self._format(code, desc)
        elif self.pending_ambiguous == "tower":
            if re.search(r"\b(3|third|three|tower\s*3)\b", text):
                self.pending_ambiguous = None
                code, desc = self._target_code("tower3")
                return self._format(code, desc)
            if re.search(r"\b(2|second|two|tower\s*2)\b", text):
                self.pending_ambiguous = None
                code, desc = self._target_code("tower2")
                return self._format(code, desc)
            if re.search(r"\b(1|first|one|tower\s*1)\b", text):
                self.pending_ambiguous = None
                code, desc = self._target_code("tower1")
                return self._format(code, desc)

        # Multi-waypoint showcase preset
        if ("solar" in text and "car" in text) or "showcase" in text or "full demo" in text:
            self.pending_ambiguous = None
            code = (
                "aw.takeoff()\n"
                "pos = aw.get_drone_position()\n"
                "aw.fly_to([pos[0], pos[1], 12.0])\n"
                'solar = aw.get_position("solarpanels")\n'
                "aw.fly_safe([solar[0], solar[1], max(solar[2] + 12.0, 10.0)])\n"
                'car = aw.get_position("car")\n'
                "aw.fly_safe([car[0], car[1], max(car[2] + 8.0, 8.0)])"
            )
            return self._format(
                code,
                "Takes off, climbs to 12m, inspects `solarpanels`, then safely flies to `car`.",
            )

        # Ambiguity checks before specific target matching
        if re.search(r"\bturbines?\b", text) and not re.search(
            r"\b(turbine\s*[12]|first|second|[12])\b", text
        ):
            self.pending_ambiguous = "turbine"
            return (
                "Question - There are two turbines in the scene (turbine1 and turbine2). "
                "Which turbine would you like me to fly to?"
            )

        if re.search(r"\btowers?\b", text) and not re.search(
            r"\b(tower\s*[123]|first|second|third|[123])\b", text
        ):
            self.pending_ambiguous = "tower"
            return (
                "Question - There are three towers in the scene (tower1, tower2, and tower3). "
                "Which tower would you like me to inspect?"
            )

        # Specific targets (word-boundary matching avoids false hits like "carefully" -> "car")
        target_patterns = [
            ("turbine1", r"\b(turbine\s*1|first\s+turbine)\b"),
            ("turbine2", r"\b(turbine\s*2|second\s+turbine)\b"),
            ("solarpanels", r"\b(solar\s*panels?|solar)\b"),
            ("car", r"\b(car|vehicle)\b"),
            ("crowd", r"\b(crowd|people|person)\b"),
            ("tower1", r"\b(tower\s*1|first\s+tower)\b"),
            ("tower2", r"\b(tower\s*2|second\s+tower)\b"),
            ("tower3", r"\b(tower\s*3|third\s+tower)\b"),
        ]
        for target, pattern in target_patterns:
            if re.search(pattern, text):
                self.pending_ambiguous = None
                code, desc = self._target_code(target)
                return self._format(code, desc)

        if "take off" in text or "takeoff" in text or "launch" in text:
            self.pending_ambiguous = None
            alt = max(self._extract_number(text, default=6.0), 3.0)
            code = (
                "aw.takeoff()\n"
                "pos = aw.get_drone_position()\n"
                f"aw.fly_to([pos[0], pos[1], max(pos[2], {alt})])"
            )
            return self._format(code, f"Takes off and climbs to a safe {alt:.1f}m hover altitude.")

        if "land" in text or "touch down" in text:
            self.pending_ambiguous = None
            return self._format("aw.land()", "Lands the drone safely at its current position.")

        if "home" in text or "origin" in text or "return" in text:
            self.pending_ambiguous = None
            return self._format(
                "aw.fly_safe([0.0, 0.0, 10.0])",
                "Returns safely to the launch origin [0, 0, 10] with obstacle avoidance.",
            )

        if "distance" in text or "clearance" in text or "sensor" in text:
            self.pending_ambiguous = None
            return self._format(
                'dist = aw.get_distance()\nprint(f"Front obstacle clearance: {dist:.2f} m")',
                "Reads current forward obstacle distance from the DepthPlanar sensor.",
            )

        if "circle" in text or "orbit" in text:
            self.pending_ambiguous = None
            radius = self._extract_number(text, default=12.0)
            code = (
                "center = aw.get_drone_position()\n"
                f"radius = {radius}\n"
                "points = [\n"
                "    [center[0] + radius * math.cos(math.radians(a)), "
                "center[1] + radius * math.sin(math.radians(a)), max(center[2], 10.0)]\n"
                "    for a in range(0, 361, 30)\n"
                "]\n"
                "aw.fly_path(points)"
            )
            return self._format(code, f"Flies a {radius:.1f}m-radius horizontal inspection orbit.")

        if any(k in text for k in ("rotate", "turn", "yaw")):
            self.pending_ambiguous = None
            deg = self._extract_number(text, default=90.0)
            if "left" in text or "counter" in text:
                deg = -abs(deg)
            return self._format(
                f"aw.set_yaw(aw.get_yaw() + ({deg}))",
                f"Rotates drone yaw by {deg:.1f} degrees.",
            )

        if any(k in text for k in ("up", "climb", "higher")):
            self.pending_ambiguous = None
            d = self._extract_number(text, default=10.0)
            return self._format(
                f"pos = aw.get_drone_position()\naw.fly_to([pos[0], pos[1], pos[2] + {d}])",
                f"Climbs {d:.1f} meters upward.",
            )

        if any(k in text for k in ("down", "lower", "descend")):
            self.pending_ambiguous = None
            d = self._extract_number(text, default=5.0)
            return self._format(
                f"pos = aw.get_drone_position()\naw.fly_to([pos[0], pos[1], max(pos[2] - {d}, 2.0)])",
                f"Descends {d:.1f} meters while staying above 2m.",
            )

        if "forward" in text or "ahead" in text:
            self.pending_ambiguous = None
            d = self._extract_number(text, default=15.0)
            return self._format(
                f"pos = aw.get_drone_position()\naw.fly_safe([pos[0] + {d}, pos[1], max(pos[2], 6.0)])",
                f"Flies {d:.1f} meters forward (+X) using obstacle avoidance.",
            )

        if "back" in text:
            self.pending_ambiguous = None
            d = self._extract_number(text, default=15.0)
            return self._format(
                f"pos = aw.get_drone_position()\naw.fly_safe([pos[0] - {d}, pos[1], max(pos[2], 6.0)])",
                f"Flies {d:.1f} meters backward (-X) using obstacle avoidance.",
            )

        if "left" in text:
            self.pending_ambiguous = None
            d = self._extract_number(text, default=10.0)
            return self._format(
                f"pos = aw.get_drone_position()\naw.fly_safe([pos[0], pos[1] - {d}, max(pos[2], 6.0)])",
                f"Flies {d:.1f} meters left (-Y) using obstacle avoidance.",
            )

        if "right" in text:
            self.pending_ambiguous = None
            d = self._extract_number(text, default=10.0)
            return self._format(
                f"pos = aw.get_drone_position()\naw.fly_safe([pos[0], pos[1] + {d}, max(pos[2], 6.0)])",
                f"Flies {d:.1f} meters right (+Y) using obstacle avoidance.",
            )

        return (
            "Question - I am running in offline preset mode and did not match that command. "
            "Try asking to take off, fly to solarpanels, turbine1, turbine2, car, crowd, tower1..3, "
            "orbit, move up/forward/left/right, or land."
        )


class DroneChatAgent:
    """Manages Gemini chat session + seamless offline preset fallback + history persistence."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        prompt_path: Path = DEFAULT_PROMPT_PATH,
        sysprompt_path: Path = DEFAULT_SYSPROMPT_PATH,
        config_path: Path = DEFAULT_CONFIG_PATH,
        offline: bool = False,
        genai_client=None,
    ):
        self.model = model
        self.prompt_path = Path(prompt_path)
        self.sysprompt_path = Path(sysprompt_path)
        self.config_path = Path(config_path)
        self.offline = offline
        self.offline_planner = OfflineDronePlanner()
        self.last_error: str | None = None

        self.sysprompt = self.sysprompt_path.read_text(encoding="utf-8")
        self.task_prompt = self.prompt_path.read_text(encoding="utf-8")

        few_shot_user = "move 10 units up"
        few_shot_model = (
            "```python\n"
            "pos = aw.get_drone_position()\n"
            "aw.fly_to([pos[0], pos[1], pos[2] + 10])\n"
            "```\n\n"
            "Gets the current drone position and climbs 10 meters along the positive Z axis."
        )
        few_shot_ack = "Understood. I am ready to generate safe AirSim drone commands or ask clarification questions."

        self.history: list[dict] = [
            {"role": "system", "text": self.sysprompt},
            {"role": "user", "text": few_shot_user},
            {"role": "assistant", "text": few_shot_model},
            {"role": "user", "text": self.task_prompt},
            {"role": "assistant", "text": few_shot_ack},
        ]

        self.client = genai_client
        self.chat = None
        if not self.offline:
            self._init_gemini(few_shot_user, few_shot_model, few_shot_ack)

    def _init_gemini(self, few_shot_user: str, few_shot_model: str, few_shot_ack: str):
        if not _GENAI_AVAILABLE:
            self.last_error = "google-genai package not installed; using offline preset mode."
            self.offline = True
            return
        if self.client is None:
            api_key = resolve_gemini_api_key(self.config_path)
            if not api_key:
                self.last_error = "GEMINI_API_KEY not found; using offline preset mode."
                self.offline = True
                return
            self.client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    timeout=15000,
                    retry_options=types.HttpRetryOptions(attempts=2),
                ),
            )

        self.chat = self._create_chat(self.model, few_shot_user, few_shot_model, few_shot_ack)

    def _create_chat(
        self, model_name: str, few_shot_user: str, few_shot_model: str, few_shot_ack: str
    ):
        seed_history = [
            types.Content(role="user", parts=[types.Part.from_text(text=few_shot_user)]),
            types.Content(role="model", parts=[types.Part.from_text(text=few_shot_model)]),
            types.Content(role="user", parts=[types.Part.from_text(text=self.task_prompt)]),
            types.Content(role="model", parts=[types.Part.from_text(text=few_shot_ack)]),
        ]
        for entry in self.history[5:]:
            role = "model" if entry["role"] == "assistant" else "user"
            seed_history.append(
                types.Content(role=role, parts=[types.Part.from_text(text=entry["text"])])
            )
        return self.client.chats.create(
            model=model_name,
            config=types.GenerateContentConfig(
                system_instruction=self.sysprompt,
                temperature=0.1,
            ),
            history=seed_history,
        )

    def set_offline(self, offline: bool):
        self.offline = offline
        if not offline and self.chat is None:
            self._init_gemini(
                "move 10 units up",
                self.history[2]["text"],
                self.history[4]["text"],
            )

    def ask(self, user_text: str) -> dict:
        source = "offline_preset" if self.offline else "gemini"
        reply_text = None
        if not self.offline and self.chat is not None:
            models_to_try = [self.model] + [
                m for m in FALLBACK_GEMINI_MODELS if m != self.model
            ]
            for idx, candidate in enumerate(models_to_try):
                try:
                    if idx > 0:
                        self.chat = self._create_chat(
                            candidate,
                            self.history[1]["text"],
                            self.history[2]["text"],
                            self.history[4]["text"],
                        )
                        self.model = candidate
                    response = self.chat.send_message(user_text)
                    text = (response.text or "").strip()
                    if not text:
                        raise RuntimeError("Empty response from Gemini.")
                    reply_text = text
                    self.last_error = None
                    break
                except Exception as exc:
                    self.last_error = str(exc)
            if reply_text is None:
                reply_text = self.offline_planner.respond(user_text)
                source = "offline_fallback"
        else:
            reply_text = self.offline_planner.respond(user_text)

        self.history.append({"role": "user", "text": user_text})
        self.history.append({"role": "assistant", "text": reply_text})
        code = extract_python_code(reply_text)
        return {
            "text": reply_text,
            "code": code,
            "is_question": is_clarification_question(reply_text) and code is None,
            "source": source,
        }


def execute_drone_code(aw, code: str) -> tuple[bool, str]:
    """Execute extracted drone Python snippet safely in a bounded namespace."""
    import contextlib
    import io

    stdout_buf = io.StringIO()
    ns = {"aw": aw, "math": math, "np": np, "time": time}
    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(code, ns, ns)
        return True, stdout_buf.getvalue().strip()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main():
    parser = argparse.ArgumentParser(description="Interactive Gemini + AirSim Drone CLI")
    parser.add_argument("--prompt", type=str, default=str(DEFAULT_PROMPT_PATH))
    parser.add_argument("--sysprompt", type=str, default=str(DEFAULT_SYSPROMPT_PATH))
    parser.add_argument("--testname", type=str, default="showcase_cli")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--offline", action="store_true", help="Start in offline preset mode")
    args = parser.parse_args()

    print(f"Initializing DroneChatAgent ({args.model})...")
    agent = DroneChatAgent(
        model=args.model,
        prompt_path=Path(args.prompt),
        sysprompt_path=Path(args.sysprompt),
        offline=args.offline,
    )
    if agent.offline:
        print(colors.YELLOW + f"[Mode: OFFLINE PRESET] {agent.last_error or ''}" + colors.ENDC)
    else:
        print(colors.GREEN + f"[Mode: GEMINI ONLINE] Model={args.model}" + colors.ENDC)

    print("Initializing AirSim...")
    aw = AirSimWrapper()
    print(colors.GREEN + "AirSim connected!" + colors.ENDC)
    print(
        "Commands: type natural language, or !status, !offline, !online, !clear, !quit / !exit"
    )

    aw.client.startRecording()
    try:
        while True:
            try:
                question = input(colors.YELLOW + "AirSim> " + colors.ENDC).strip()
            except EOFError:
                question = "!exit"

            if not question:
                continue
            if question in ("!quit", "!exit"):
                saved = write_chat_history(args.model, args.testname, agent.history)
                print(colors.CYAN + f"Saved chat history to {saved}" + colors.ENDC)
                break
            if question == "!clear":
                os.system("cls" if os.name == "nt" else "clear")
                continue
            if question == "!offline":
                agent.set_offline(True)
                print(colors.YELLOW + "Switched to OFFLINE preset mode." + colors.ENDC)
                continue
            if question == "!online":
                agent.set_offline(False)
                print(
                    colors.YELLOW + f"Offline={agent.offline} ({agent.last_error or 'Gemini ready'})" + colors.ENDC
                )
                continue
            if question == "!status":
                snap = aw.refresh_snapshot()
                print(colors.CYAN + f"Telemetry: {snap}" + colors.ENDC)
                continue

            result = agent.ask(question)
            tag = "[OFFLINE FALLBACK]" if result["source"] != "gemini" else "[GEMINI]"
            print(f"\n{colors.CYAN}{tag}{colors.ENDC} {result['text']}\n")

            if result["code"]:
                print("Executing generated code in AirSim...")
                ok, out = execute_drone_code(aw, result["code"])
                if out:
                    print(out)
                if ok:
                    print(colors.GREEN + "Done!\n" + colors.ENDC)
                else:
                    print(colors.RED + "Execution error above.\n" + colors.ENDC)
    finally:
        try:
            aw.client.stopRecording()
        except Exception:
            pass


if __name__ == "__main__":
    main()
