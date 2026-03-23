"""
Flask web server for the TTS → Edge Impulse Pipeline.

Routes
------
GET  /          Serve index.html
POST /run       Validate input, save recordings, start pipeline thread
GET  /stream    SSE — live log stream until pipeline finishes
GET  /download  Download exported_model/model.tflite
"""

import builtins
import os
import queue
import re
import threading

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, send_file

from modules import TTSGenerator, DatasetBuilder, EdgeImpulseClient, ModelExporter
from modules.tts_generator import _safe_filename

load_dotenv()

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global pipeline state
# ---------------------------------------------------------------------------

log_queue: queue.Queue = queue.Queue()
pipeline = {"running": False, "done": False, "error": None}
lock = threading.Lock()

DATASET_DIR = os.getenv("DATASET_DIR", "dataset")
EXPORTED_DIR = "exported_model"
MODEL_PATH = os.path.join(EXPORTED_DIR, "model.tflite")
MIN_RECORDINGS = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict | None:
    api_key = os.getenv("EDGE_IMPULSE_API_KEY", "")
    project_id = os.getenv("EDGE_IMPULSE_PROJECT_ID", "")
    if not api_key or not project_id:
        return None
    return {
        "api_key": api_key,
        "project_id": project_id,
        "dataset_dir": DATASET_DIR,
        "sample_rate": int(os.getenv("SAMPLE_RATE", "16000")),
        "duration": float(os.getenv("SAMPLE_DURATION", "1.5")),
        "samples_per_label": int(os.getenv("TTS_SAMPLES_PER_LABEL", "20")),
        "tts_volume": float(os.getenv("TTS_VOLUME", "1.0")),
    }


def _patched_print(log_q: queue.Queue):
    """Return a replacement for builtins.print that also pushes to log_q."""
    real_print = builtins.print

    def _print(*args, **kwargs):
        real_print(*args, **kwargs)
        msg = " ".join(str(a) for a in args)
        log_q.put(msg)

    return _print


def _safe_name(label: str) -> str:
    """Return filesystem-safe ASCII name for a label."""
    return _safe_filename(label)


# ---------------------------------------------------------------------------
# Pipeline thread
# ---------------------------------------------------------------------------

def _run_pipeline(
    labels: list[str],
    mode: str,
    recordings: dict[str, list[str]],   # label → list of temp webm paths (record mode)
    training_cycles: int,
    learning_rate: float,
    batch_size: int,
    autotune: bool,
    samples_per_label: int,
    duration: float,
):
    cfg = _load_config()
    if cfg is None:
        log_queue.put("[ERROR] EDGE_IMPULSE_API_KEY / EDGE_IMPULSE_PROJECT_ID not set in .env")
        log_queue.put("__DONE__")
        with lock:
            pipeline["running"] = False
            pipeline["error"] = "Missing API credentials"
        return

    cfg["samples_per_label"] = samples_per_label
    cfg["duration"] = duration

    # Monkey-patch print so every module's output reaches the SSE stream
    original_print = builtins.print
    builtins.print = _patched_print(log_queue)

    try:
        tts = TTSGenerator(
            output_dir=cfg["dataset_dir"],
            sample_rate=cfg["sample_rate"],
            duration=cfg["duration"],
            samples_per_label=cfg["samples_per_label"],
            tts_volume=cfg["tts_volume"],
        )

        # ---- Step 1: Audio ------------------------------------------------
        print("=== Step 1/4: Processing audio ===")
        if mode == "tts":
            tts.generate(labels)
        else:
            for label in labels:
                paths = recordings.get(label, [])
                tts.generate_from_recordings(label, paths)

        # ---- Step 2: Split ------------------------------------------------
        print("=== Step 2/4: Splitting dataset (80/20 train/test) ===")
        builder = DatasetBuilder(dataset_dir=cfg["dataset_dir"])
        split = builder.build(only_labels=[_safe_name(l) for l in labels])

        # ---- Step 3: Upload + Train ----------------------------------------
        client = EdgeImpulseClient(api_key=cfg["api_key"], project_id=cfg["project_id"])

        print("=== Step 3a/4: Clearing all existing Edge Impulse data ===")
        client.clear_all_project_data()

        print("=== Step 3b/4: Uploading dataset to Edge Impulse ===")
        client.upload_dataset(split)

        if autotune:
            print("=== Step 3c/4: Running DSP auto-tune ===")
            client.run_dsp_autotune()

        print("=== Step 3d/4: Starting training job ===")
        job_id = client.start_training(
            training_cycles=training_cycles,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )
        success = client.wait_for_training(job_id)
        if not success:
            print("[ERROR] Training failed. Check Edge Impulse Studio for details.")
            with lock:
                pipeline["running"] = False
                pipeline["error"] = "Training failed"
            return

        # ---- Step 4: Export -----------------------------------------------
        print("=== Step 4/4: Downloading trained TFLite model ===")
        exporter = ModelExporter(
            api_key=cfg["api_key"],
            project_id=cfg["project_id"],
            output_dir=EXPORTED_DIR,
        )
        exporter.download(labels)

        print("=== Pipeline complete! Model ready for download. ===")
        with lock:
            pipeline["running"] = False
            pipeline["done"] = True

    except Exception as exc:
        print(f"[ERROR] {exc}")
        with lock:
            pipeline["running"] = False
            pipeline["error"] = str(exc)
    finally:
        builtins.print = original_print
        # Clean up temp webm files
        for paths in recordings.values():
            for p in paths:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except OSError:
                    pass
        log_queue.put("__DONE__")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run():
    with lock:
        if pipeline["running"]:
            return jsonify({"error": "Pipeline already running"}), 409
        pipeline["running"] = True
        pipeline["done"] = False
        pipeline["error"] = None

    # Drain old log messages
    while not log_queue.empty():
        try:
            log_queue.get_nowait()
        except queue.Empty:
            break

    mode = request.form.get("mode", "tts")

    # Collect labels
    labels = []
    i = 0
    while True:
        label = request.form.get(f"label_{i}")
        if label is None:
            break
        labels.append(label.strip())
        i += 1

    if not labels:
        with lock:
            pipeline["running"] = False
        return jsonify({"error": "No labels provided"}), 400

    # Settings
    try:
        samples_per_label = int(request.form.get("samples_per_label", 20))
        duration = float(request.form.get("duration", 1.5))
        training_cycles = int(request.form.get("training_cycles", 100))
        learning_rate = float(request.form.get("learning_rate", 0.0005))
        batch_size = int(request.form.get("batch_size", 32))
        autotune = request.form.get("autotune", "false").lower() == "true"
    except ValueError as e:
        with lock:
            pipeline["running"] = False
        return jsonify({"error": f"Invalid settings: {e}"}), 400

    # Save recordings (record mode)
    recordings: dict[str, list[str]] = {}
    if mode == "record":
        for idx, label in enumerate(labels):
            recordings[label] = []
            j = 0
            while True:
                file = request.files.get(f"recordings_{idx}_{j}")
                if file is None:
                    break
                safe = _safe_name(label)
                dest_dir = os.path.join(DATASET_DIR, safe)
                os.makedirs(dest_dir, exist_ok=True)
                dest = os.path.join(dest_dir, f"rec_{j}.webm")
                file.save(dest)
                recordings[label].append(dest)
                j += 1

        # Validate: every label needs at least MIN_RECORDINGS
        for label in labels:
            if len(recordings.get(label, [])) < MIN_RECORDINGS:
                with lock:
                    pipeline["running"] = False
                return jsonify({
                    "error": f"Label '{label}' needs at least {MIN_RECORDINGS} recordings"
                }), 400

    t = threading.Thread(
        target=_run_pipeline,
        args=(
            labels, mode, recordings,
            training_cycles, learning_rate, batch_size,
            autotune, samples_per_label, duration,
        ),
        daemon=True,
    )
    t.start()
    return jsonify({"status": "started"})


@app.route("/stream")
def stream():
    def generate():
        while True:
            try:
                msg = log_queue.get(timeout=30)
            except queue.Empty:
                yield "data: [waiting...]\n\n"
                continue
            if msg == "__DONE__":
                yield "data: __DONE__\n\n"
                break
            # Escape newlines for SSE
            safe = msg.replace("\n", " ")
            yield f"data: {safe}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/download")
def download():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not found — run the pipeline first"}), 404
    return send_file(
        os.path.abspath(MODEL_PATH),
        as_attachment=True,
        download_name="model.tflite",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("TTS → Edge Impulse Web UI")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=False, host="0.0.0.0", port=5000)
