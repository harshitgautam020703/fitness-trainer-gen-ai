import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import os
import requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ── CONFIG ──────────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/chat"   # Ollama local server
OLLAMA_MODEL = "gemma2:2b"                     # Model tag after `ollama pull`
MODEL_PATH = "pose_landmarker_heavy.task"

app = Flask(__name__)
CORS(app)

state = {
    "counter": 0,
    "exercise": "Bicep Curl",
    "stage": "idle",
    "angle": 0.0,
    "session_start": time.time(),
    "rep_log": [],
    "running": True
}

# ── ROUTES ───────────────────────────────────────────────────────────────────
@app.route("/")
def serve_frontend():
    return send_from_directory(".", "index.html")

@app.route("/state")
def get_state():
    return jsonify({
        "counter": state["counter"],
        "exercise": state["exercise"],
        "stage": state["stage"],
        "angle": state["angle"],
        "elapsed": time.time() - state["session_start"],
        "rep_log": state["rep_log"][-8:]
    })

@app.route("/set_exercise", methods=["POST"])
def set_exercise():
    state["exercise"] = request.json.get("exercise", "Bicep Curl")
    return jsonify({"ok": True})

@app.route("/reset", methods=["POST"])
def reset():
    state.update({
        "counter": 0,
        "stage": "idle",
        "angle": 0.0,
        "rep_log": [],
        "session_start": time.time()
    })
    return jsonify({"ok": True})

@app.route("/chat", methods=["POST"])
def chat():
    """Ollama Llama 3.1 8B powered dietary coaching endpoint (fully local, no API key)."""
    data = request.json or {}
    user_msg = data.get("message", "")
    history = data.get("history", [])   # list of {"role": "user"/"assistant", "content": "..."}
    profile = data.get("profile", {})

    system_prompt = (
        "You are APEX Coach, an expert AI fitness nutritionist. "
        f"User profile: {profile}. "
        "Give concise, practical, personalized nutrition advice. "
        "Use emojis sparingly. Format clearly with line breaks."
    )

    # Build messages list for Ollama (OpenAI-style format)
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history if provided
    for turn in history:
        messages.append({
            "role": turn.get("role", "user"),
            "content": turn.get("content", "")
        })

    # Add current user message if not already in history
    if not history or history[-1].get("role") != "user":
        messages.append({"role": "user", "content": user_msg})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,        # Get full response at once
        "options": {
            "temperature": 0.7,
            "num_predict": 512  # Max tokens in reply
        }
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
        resp.raise_for_status()
        reply = resp.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        reply = (
            "⚠️ Ollama is not running.\n"
            "Start it with: ollama serve\n"
            "Then pull the model: ollama pull llama3.1:8b"
        )
    except requests.exceptions.Timeout:
        reply = "⚠️ Request timed out. The model may still be loading — try again in a moment."
    except Exception as e:
        reply = f"⚠️ Error: {e}"

    return jsonify({"reply": reply})

# ── ANGLE MATH ───────────────────────────────────────────────────────────────
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = (np.arctan2(c[1] - b[1], c[0] - b[0]) -
               np.arctan2(a[1] - b[1], a[0] - b[0]))
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# ── POSE ENGINE ──────────────────────────────────────────────────────────────
def run_pose_engine():
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO
    )
    cap = cv2.VideoCapture(0)

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            ts_ms = int(time.time() * 1000)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            result = landmarker.detect_for_video(mp_image, ts_ms)

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                ex = state["exercise"]

                # ── Landmark shortcuts ──
                sR = [lm[12].x, lm[12].y]
                eR = [lm[14].x, lm[14].y]
                wR = [lm[16].x, lm[16].y]
                sL = [lm[11].x, lm[11].y]
                eL = [lm[13].x, lm[13].y]
                wL = [lm[15].x, lm[15].y]
                h  = [lm[23].x, lm[23].y]
                k  = [lm[25].x, lm[25].y]
                an = [lm[27].x, lm[27].y]

                angle = 0.0

                if ex == "Bicep Curl":
                    angle = calculate_angle(sL, eL, wL)
                    if angle > 160:
                        state["stage"] = "down"
                    if angle < 40 and state["stage"] == "down":
                        state["stage"] = "up"
                        state["counter"] += 1

                elif ex == "Lateral Raise":
                    angle = calculate_angle(h, sL, eL)
                    if angle < 30:
                        state["stage"] = "down"
                    if angle > 85 and state["stage"] == "down":
                        state["stage"] = "up"
                        state["counter"] += 1

                elif ex == "Squats":
                    angle = calculate_angle(h, k, an)
                    if angle > 165:
                        state["stage"] = "up"
                    if angle < 100 and state["stage"] == "up":
                        state["stage"] = "down"
                        state["counter"] += 1

                elif ex in ["Overhead Press", "Bench Press", "Tricep Extension"]:
                    angle = calculate_angle(sL, eL, wL)
                    if angle < 70:
                        state["stage"] = "down"
                    if angle > 155 and state["stage"] == "down":
                        state["stage"] = "up"
                        state["counter"] += 1

                elif ex == "Pull Ups":
                    diff = wL[1] - sL[1]
                    angle = abs(diff) * 180
                    if diff > 0.05:
                        state["stage"] = "down"
                    if diff < -0.05 and state["stage"] == "down":
                        state["stage"] = "up"
                        state["counter"] += 1

                state["angle"] = round(angle, 1)

                is_new_rep = (
                    state["stage"] == "up" and
                    (not state["rep_log"] or
                     state["rep_log"][-1]["rep"] != state["counter"])
                )
                if is_new_rep and state["counter"] > 0:
                    state["rep_log"].append({
                        "ex": ex,
                        "rep": state["counter"],
                        "ts": round(time.time() - state["session_start"], 1)
                    })

                # ── Draw skeleton overlay ──
                h_px, w_px = frame.shape[:2]
                def to_px(pt): return (int(pt[0] * w_px), int(pt[1] * h_px))

                pts = {
                    "sL": to_px(sL), "eL": to_px(eL), "wL": to_px(wL),
                    "sR": to_px(sR), "eR": to_px(eR), "wR": to_px(wR),
                    "h":  to_px(h),  "k":  to_px(k),  "an": to_px(an),
                }

                skeleton = [
                    ("sL","eL"), ("eL","wL"), ("sR","eR"), ("eR","wR"),
                    ("sL","sR"), ("sL","h"), ("sR","h"), ("h","k"), ("k","an")
                ]
                for a_key, b_key in skeleton:
                    cv2.line(frame, pts[a_key], pts[b_key], (0, 230, 160), 2)
                for pt in pts.values():
                    cv2.circle(frame, pt, 5, (255, 80, 80), -1)

                cv2.putText(frame, f"{round(angle, 1)}", to_px(eL),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.rectangle(frame, (0, 0), (250, 60), (10, 10, 20), -1)
                cv2.putText(frame, f"REPS  STAGE", (10, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 130), 1)
                cv2.putText(frame,
                            f"{state['counter']}  {state['stage'].upper()}",
                            (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0, 230, 160), 2)

            cv2.imshow("APEX Pose Engine", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=run_pose_engine, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)