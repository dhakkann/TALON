"""AR511 Real-Time Mission & Gemini Pipeline Monitor.
Displays live incoming commands, Gemini model invocations, and drone actions.
"""

import os
import sys
import time
import json

LOG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "mission_audit.log"))


def main():
    print("=" * 70)
    print(" AR511 Mission & Gemini Pipeline Live Monitor Active")
    print(f" Watching: {LOG_PATH}")
    print(" Press Ctrl+C to stop.")
    print("=" * 70 + "\n")

    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            pass

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        # Seek to end
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.2)
                continue

            try:
                evt = json.loads(line.strip())
                ts = evt.get("timestamp", "")
                cat = evt.get("category", "")

                if cat == "MISSION_PLANNED":
                    eng = evt.get("engine", "")
                    mod = evt.get("model", "")
                    instr = evt.get("instruction", "")
                    subgoals = evt.get("subgoals", [])
                    print(f"[{ts}] 🧠 MISSION PLANNED ({eng} | {mod})")
                    print(f"    Instruction : \"{instr}\"")
                    print(f"    Subgoals    : {' -> '.join(subgoals)}")
                    print(f"    Rationale   : {evt.get('rationale')}\n")

                elif cat == "OPERATOR_INTERRUPTION":
                    eng = evt.get("engine", "")
                    cmd = evt.get("command", "")
                    intent = evt.get("intent", "")
                    tgt = evt.get("target", "")
                    reply = evt.get("reply", "")
                    print(f"[{ts}] ⚡ OPERATOR INTERRUPTION ({eng})")
                    print(f"    Command : \"{cmd}\"")
                    print(f"    Intent  : {intent} (Target: {tgt})")
                    print(f"    Feedback: \"{reply}\"\n")

                elif cat == "VOICE_TRANSCRIBED":
                    src = evt.get("source", "")
                    txt = evt.get("transcript", "")
                    lat = evt.get("latency_ms", 0.0)
                    print(f"[{ts}] 🎙️ VOICE TRANSCRIBED ({src} in {lat:.1f}ms)")
                    print(f"    Spoken Text : \"{txt}\"\n")

                elif cat == "MISSION_EVENT":
                    etype = evt.get("event_type", "")
                    msg = evt.get("message", "")
                    spoken = evt.get("spoken_text", "")
                    print(f"[{ts}] 🚁 [{etype}] {msg}")
                    if spoken:
                        print(f"    Audio TTS: \"{spoken}\"")
                    print()

                else:
                    print(f"[{ts}] [{cat}] {evt}\n")

            except Exception:
                print(line.strip())


if __name__ == "__main__":
    main()
