import hashlib
import re
import base64
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, make_response, render_template, request, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename


ROOT = Path(__file__).resolve().parent
JOB_ROOT = ROOT / "web_jobs"
MAX_UPLOAD_SIZE = 500 * 1024 * 1024
MAX_IMAGE_COUNT = 40
JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
pipeline_lock = threading.Lock()
STATIC_ASSET_VERSION = hashlib.sha256(
    (ROOT / "static" / "app.js").read_bytes()
    + (ROOT / "static" / "styles.css").read_bytes()
).hexdigest()[:12]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE


def job_directory(job_id):
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise ValueError("Invalid job ID.")
    return JOB_ROOT / job_id


def run_pipeline(command):
    if not pipeline_lock.acquire(blocking=False):
        return jsonify(error="Another conversion or registration is already running."), 409
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=6 * 60 * 60,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return jsonify(error=f"Processing timed out: {error}"), 504
    finally:
        pipeline_lock.release()

    if result.returncode:
        details = (result.stderr or result.stdout).strip()
        return jsonify(error=details or "The processing command failed."), 500
    return jsonify(message=result.stdout.strip())


@app.get("/")
def index():
    response = make_response(
        render_template("index.html", asset_version=STATIC_ASSET_VERSION)
    )
    response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/api/health")
def health():
    return jsonify(ready=True)


@app.post("/api/upload")
def upload_images():
    files = request.files.getlist("images")
    if not files or all(not file.filename for file in files):
        return jsonify(error="Choose one or more JPG images first."), 400
    files = [file for file in files if file.filename]
    if len(files) < 2:
        return jsonify(error="Choose at least two JPG images to register a room."), 400
    if len(files) > MAX_IMAGE_COUNT:
        return jsonify(error=f"Upload no more than {MAX_IMAGE_COUNT} images at once."), 400

    job_id = uuid.uuid4().hex
    upload_directory = job_directory(job_id) / "images"
    upload_directory.mkdir(parents=True, exist_ok=False)
    saved_files = []
    try:
        for index, file in enumerate(files, start=1):
            original_name = secure_filename(file.filename)
            if Path(original_name).suffix.lower() not in {".jpg", ".jpeg"}:
                raise ValueError(f"{file.filename} is not a JPG image.")
            try:
                image = Image.open(file.stream)
                if image.format != "JPEG":
                    raise ValueError(f"{file.filename} is not a JPG image.")
                image.verify()
            except Exception as error:
                if isinstance(error, ValueError):
                    raise
                raise ValueError(f"{file.filename} is not a readable JPG image.") from error
            file.stream.seek(0)
            image_name = f"{index:03d}_{original_name}"
            file.save(upload_directory / image_name)
            saved_files.append(image_name)
    except ValueError as error:
        for path in upload_directory.iterdir():
            path.unlink()
        upload_directory.rmdir()
        upload_directory.parent.rmdir()
        return jsonify(error=str(error)), 400

    return jsonify(jobId=job_id, images=saved_files), 201


@app.post("/api/jobs/<job_id>/convert")
def convert_images(job_id):
    try:
        paths = job_directory(job_id)
    except ValueError as error:
        return jsonify(error=str(error)), 400
    input_directory = paths / "images"
    if not input_directory.is_dir():
        return jsonify(error="Upload images before converting them."), 404

    point_cloud_directory = paths / "point_clouds"
    command = [
        sys.executable,
        str(ROOT / "depth_to_pointcloud.py"),
        "--input-dir",
        str(input_directory),
        "--output-dir",
        str(point_cloud_directory),
    ]
    response = run_pipeline(command)
    if isinstance(response, tuple):
        return response
    files = sorted(point_cloud_directory.glob("*.ply"))
    if not files:
        return jsonify(error="The conversion completed without producing point clouds."), 500
    return jsonify(message="Images converted to point clouds.", count=len(files))


@app.post("/api/jobs/<job_id>/register")
def register_point_clouds(job_id):
    try:
        paths = job_directory(job_id)
    except ValueError as error:
        return jsonify(error=str(error)), 400
    point_cloud_directory = paths / "point_clouds"
    if len(list(point_cloud_directory.glob("*.ply"))) < 2:
        return jsonify(error="Convert at least two images before registering the point clouds."), 400

    output_directory = paths / "results"
    command = [
        sys.executable,
        str(ROOT / "register_and_floorplan.py"),
        "--input-dir",
        str(point_cloud_directory),
        "--output-dir",
        str(output_directory),
        "--no-visualization",
    ]
    response = run_pipeline(command)
    if isinstance(response, tuple):
        return response

    files = [
        {"name": name, "url": f"/api/jobs/{job_id}/files/{name}"}
        for name in ("complete_room_merged.ply", "extracted_floor_plan.ply", "floor_plan_preview.png")
        if (output_directory / name).is_file()
    ]
    if len(files) != 3:
        return jsonify(error="Registration finished without producing all expected output files."), 500
    return jsonify(message="Point clouds registered and floor plan created.", files=files)


@app.get("/api/jobs/<job_id>/files/<filename>")
def download_result(job_id, filename):
    if filename not in {
        "complete_room_merged.ply",
        "extracted_floor_plan.ply",
        "floor_plan_preview.png",
    }:
        return jsonify(error="Unknown output file."), 404
    try:
        output_directory = job_directory(job_id) / "results"
    except ValueError as error:
        return jsonify(error=str(error)), 400
    return send_from_directory(output_directory, filename, as_attachment=True)


@app.get("/api/jobs/<job_id>/preview")
def preview_point_cloud(job_id):
    try:
        output_directory = job_directory(job_id) / "results"
    except ValueError as error:
        return jsonify(error=str(error)), 400

    merged_path = output_directory / "complete_room_merged.ply"
    if not merged_path.is_file():
        return jsonify(error="Register the scans before previewing the merged point cloud."), 404

    import numpy as np
    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(merged_path))
    if cloud.is_empty():
        return jsonify(error="The merged point cloud is empty or could not be read."), 500
    cloud = cloud.voxel_down_sample(voxel_size=0.02)
    points = np.asarray(cloud.points)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if not len(points):
        return jsonify(error="The merged point cloud contains no finite points."), 500

    colors = np.asarray(cloud.colors)
    if len(colors) != len(finite):
        colors = np.tile(np.array([[0.79, 0.93, 0.42]]), (len(finite), 1))
    colors = colors[finite]
    if len(points) > 25000:
        indices = np.linspace(0, len(points) - 1, 25000, dtype=np.intp)
        points = points[indices]
        colors = colors[indices]

    bounds = np.stack((points.min(axis=0), points.max(axis=0)))
    center = bounds.mean(axis=0)
    scale = float(np.max(bounds[1] - bounds[0]))
    if not np.isfinite(scale) or scale <= 0:
        return jsonify(error="The merged point cloud has no measurable dimensions."), 500

    normalized_points = ((points - center) / scale).astype("<f4", copy=False)
    packed_colors = np.clip(colors * 255, 0, 255).astype(np.uint8, copy=False)
    payload = jsonify(
        positions=base64.b64encode(normalized_points.tobytes()).decode("ascii"),
        colors=base64.b64encode(packed_colors.tobytes()).decode("ascii"),
        count=len(points),
        dimensions=np.round(bounds[1] - bounds[0], 3).tolist(),
    )
    payload.headers["Cache-Control"] = "no-store"
    return payload


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify(error="The upload is too large. The total limit is 500 MB."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
