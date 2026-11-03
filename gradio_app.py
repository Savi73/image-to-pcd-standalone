import os
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"

import gradio as gr
import spaces
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename


ROOT = Path(__file__).resolve().parent
JOB_ROOT = ROOT / "web_jobs"
MAX_UPLOAD_SIZE = 500 * 1024 * 1024
MAX_IMAGE_COUNT = 40
pipeline_lock = threading.Lock()


def resolve_encoder_checkpoint():
    checkpoint = Path(
        os.environ.get(
            "DEPTH_ENCODER_CHECKPOINT_PATH",
            str(ROOT / "checkpoints" / "depth_anything_vitl14.pth"),
        )
    ).expanduser()
    if checkpoint.is_file():
        os.environ["DEPTH_ENCODER_CHECKPOINT_PATH"] = str(checkpoint)
        return checkpoint

    model_repo = os.environ.get(
        "DEPTH_ENCODER_MODEL_REPO", os.environ.get("DEPTH_MODEL_REPO")
    )
    if not model_repo:
        raise FileNotFoundError(
            f"Encoder checkpoint not found at {checkpoint}. Set "
            "DEPTH_ENCODER_MODEL_REPO or DEPTH_MODEL_REPO to a Hugging Face model "
            "repository, or set DEPTH_ENCODER_CHECKPOINT_PATH to a local file."
        )

    filename = os.environ.get("DEPTH_ENCODER_FILENAME", checkpoint.name)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(
        hf_hub_download(
            repo_id=model_repo,
            filename=filename,
            token=os.environ.get("HF_TOKEN"),
            local_dir=str(checkpoint.parent),
        )
    )
    os.environ["DEPTH_ENCODER_CHECKPOINT_PATH"] = str(checkpoint)
    return checkpoint


def validate_images(image_files):
    if not image_files:
        raise gr.Error("Upload at least two JPG room photos.")
    if len(image_files) < 2:
        raise gr.Error("Upload at least two JPG room photos to register a room.")
    if len(image_files) > MAX_IMAGE_COUNT:
        raise gr.Error(f"Upload no more than {MAX_IMAGE_COUNT} images at once.")

    total_size = sum(Path(file_path).stat().st_size for file_path in image_files)
    if total_size > MAX_UPLOAD_SIZE:
        raise gr.Error("The total upload limit is 500 MB.")

    for file_path in image_files:
        if Path(file_path).suffix.lower() not in {".jpg", ".jpeg"}:
            raise gr.Error(f"{Path(file_path).name} is not a JPG image.")
        try:
            with Image.open(file_path) as image:
                if image.format != "JPEG":
                    raise gr.Error(f"{Path(file_path).name} is not a JPG image.")
                image.verify()
        except (UnidentifiedImageError, OSError) as error:
            raise gr.Error(f"{Path(file_path).name} is not a readable JPG image.") from error


def run_command(command):
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
        raise gr.Error("Room scan timed out after six hours.") from error
    if result.returncode:
        details = (result.stderr or result.stdout).strip()
        raise gr.Error(details[-4000:] or "The processing command failed.")


@spaces.GPU(duration=600)
def create_room_scan(image_files):
    validate_images(image_files)
    if not pipeline_lock.acquire(blocking=False):
        raise gr.Error("Another room scan is already running. Please try again later.")

    try:
        resolve_encoder_checkpoint()
        job_directory = JOB_ROOT / f"hf_gradio_{uuid.uuid4().hex}"
        image_directory = job_directory / "images"
        point_cloud_directory = job_directory / "point_clouds"
        output_directory = job_directory / "results"
        image_directory.mkdir(parents=True, exist_ok=False)

        for index, file_path in enumerate(image_files, start=1):
            image_stem = secure_filename(Path(file_path).stem) or "image"
            image_name = f"{image_stem}.jpg"
            shutil.copyfile(file_path, image_directory / f"{index:03d}_{image_name}")

        run_command(
            [
                sys.executable,
                str(ROOT / "depth_to_pointcloud.py"),
                "--input-dir",
                str(image_directory),
                "--output-dir",
                str(point_cloud_directory),
            ]
        )
        point_clouds = list(point_cloud_directory.glob("*.ply"))
        if len(point_clouds) < 2:
            raise gr.Error("Depth conversion completed without producing two point clouds.")

        run_command(
            [
                sys.executable,
                str(ROOT / "register_and_floorplan.py"),
                "--input-dir",
                str(point_cloud_directory),
                "--output-dir",
                str(output_directory),
                "--no-visualization",
            ]
        )

        merged_cloud = output_directory / "complete_room_merged.ply"
        floor_plan = output_directory / "extracted_floor_plan.ply"
        floor_plan_preview = output_directory / "floor_plan_preview.png"
        missing_outputs = [
            path.name for path in (merged_cloud, floor_plan, floor_plan_preview)
            if not path.is_file()
        ]
        if missing_outputs:
            raise gr.Error(
                "Registration did not produce the expected output files: "
                + ", ".join(missing_outputs)
            )

        return (
            "Room scan complete. Download the merged point cloud and floor plan below.",
            str(merged_cloud),
            str(floor_plan),
            str(floor_plan_preview),
        )
    finally:
        pipeline_lock.release()


with gr.Blocks(title="Roomscan | Photos to Point Cloud") as demo:
    gr.Markdown(
        """
        # Roomscan
        Turn **2–40 JPG photos** of a room into a registered 3D point cloud and
        floor plan. For best registration, take overlapping photos as you move
        around the room.

        The first scan downloads the model checkpoints and may take several
        minutes. Uploaded photos and results are stored temporarily while this
        Space is running; do not upload sensitive images.
        """
    )
    images = gr.File(
        label="Room photos (JPG)",
        file_count="multiple",
        file_types=[".jpg", ".jpeg"],
        type="filepath",
    )
    submit = gr.Button("Create point cloud and floor plan", variant="primary")
    status = gr.Markdown()
    with gr.Row():
        merged_cloud = gr.File(label="Merged room point cloud (PLY)")
        floor_plan = gr.File(label="Floor plan point cloud (PLY)")
    floor_plan_preview = gr.Image(label="Floor plan preview")

    submit.click(
        fn=create_room_scan,
        inputs=images,
        outputs=[status, merged_cloud, floor_plan, floor_plan_preview],
    )

demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        max_file_size="500mb",
        show_error=True,
    )
