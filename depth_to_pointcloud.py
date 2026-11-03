import argparse
import os
import glob
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import open3d as o3d
from tqdm import tqdm
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

NYU_DATA = False
DATASET = 'nyu'  # For INDOOR
# DATASET = 'kitti'  # For OUTDOOR


def process_images(model, input_dir, output_dir, camera_matrix):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(
        glob.glob(os.path.join(input_dir, '*.jpg'))
        + glob.glob(os.path.join(input_dir, '*.jpeg'))
        + glob.glob(os.path.join(input_dir, '*.png'))
    )
    if not image_paths:
        raise ValueError(f"No supported images found in {input_dir}")

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    for image_path in tqdm(image_paths, desc="Processing Images"):
        color_image = Image.open(image_path).convert('RGB')
        width, height = color_image.size
        image_tensor = transforms.ToTensor()(color_image).unsqueeze(0).to(
            'cuda' if torch.cuda.is_available() else 'cpu')

        pred = model(image_tensor, dataset=DATASET)
        if isinstance(pred, dict):
            pred = pred.get('metric_depth', pred.get('out'))
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        if pred is None:
            raise ValueError(f"The depth model returned no prediction for {image_path}")
        pred = pred.squeeze().detach().cpu().numpy()

        resized_pred = Image.fromarray(pred).resize((width, height), Image.NEAREST)
        focal_length_x, focal_length_y = (fx, fy) if not NYU_DATA else ((fx + fy) / 2,) * 2
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = (x - width / 2) / focal_length_x
        y = (y - height / 2) / focal_length_y
        z = np.array(resized_pred)
        points = np.stack((x * z, y * z, z), axis=-1).reshape(-1, 3)
        colors = np.array(color_image).reshape(-1, 3) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        output_path = os.path.join(
            output_dir, os.path.splitext(os.path.basename(image_path))[0] + ".ply")
        if not o3d.io.write_point_cloud(output_path, pcd):
            raise OSError(f"Could not write point cloud: {output_path}")


def resolve_checkpoint_path(checkpoint_path):
    checkpoint = Path(checkpoint_path).expanduser()
    if checkpoint.is_file():
        return checkpoint

    model_repo = os.environ.get("DEPTH_MODEL_REPO")
    if not model_repo:
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint}. Set DEPTH_MODEL_REPO to a "
            "Hugging Face model repository or DEPTH_CHECKPOINT_PATH to a local file."
        )

    from huggingface_hub import hf_hub_download

    filename = os.environ.get("DEPTH_MODEL_FILENAME", checkpoint.name)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(
        repo_id=model_repo,
        filename=filename,
        token=os.environ.get("HF_TOKEN"),
        local_dir=str(checkpoint.parent),
    ))


def main(input_dir, output_dir, calibration_path, model_name, pretrained_resource):
    calibration_data = np.load(calibration_path)
    camera_matrix = calibration_data['Camera_matrix']
    print(f"FX: {camera_matrix[0, 0]}")
    print(f"FY: {camera_matrix[1, 1]}")
    print(f"FL (average focal length): {(camera_matrix[0, 0] + camera_matrix[1, 1]) / 2}")

    config = get_config(model_name, "eval", DATASET)
    config.pretrained_resource = pretrained_resource
    model = build_model(config).to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    process_images(model, input_dir, output_dir, camera_matrix)


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Convert images into metric point clouds.")
    parser.add_argument("--input-dir", default=str(project_dir / "my_test" / "input" / "indoor"))
    parser.add_argument("--output-dir", default=str(project_dir / "my_test" / "output" / "indoor"))
    parser.add_argument("--calibration", default=str(project_dir / "CalibrationMatrix_college_cpt.npz"))
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get(
            "DEPTH_CHECKPOINT_PATH",
            str(project_dir / "checkpoints" / "depth_anything_metric_depth_indoor.pt"),
        ))
    args = parser.parse_args()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    main(
        args.input_dir,
        args.output_dir,
        args.calibration,
        "zoedepth",
        f"local::{checkpoint_path}")
