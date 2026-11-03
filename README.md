---
title: Roomscan
emoji: 🏠
colorFrom: green
colorTo: yellow
sdk: gradio
app_file: gradio_app.py
python_version: "3.12.12"
suggested_hardware: cpu-upgrade
---

## Metric Point Cloud Creation from Single Images using Depth Anything Model

This project leverages the Depth Anything model to create metric point clouds from single images. This repository includes all necessary scripts and files to perform depth estimation and generate accurate 3D point clouds with real-world measurements.

### Requirements

------------

To install the required packages, run:
`pip install -r requirements.txt`

### Installation

------------

1. Clone this repository:
`git clone https://github.com/bohdanvodianyk/image-to-pcd.git`
`cd your-repo-name`
2. Install the necessary Python packages:
`pip install -r requirements.txt`
3. Download the model checkpoints from [Google Drive](https://drive.google.com/drive/folders/1LJRnpOhNuzZXlVE0oGzzUb7ZiXeF6f-8?usp=sharing "Google Drive") and place them in the appropriate directory within the project.

### Usage

------------

#### Calibration
Before generating point clouds, calibrate your camera using the `calibration-camera.py` script. Ensure you have a chessboard pattern printed for the calibration process.

#### Depth Estimation to Point Cloud
To convert depth maps into metric point clouds, use the `depth_to_pointcloud.py` script. Ensure your input image is correctly formatted and accessible.

#### HEIC to PNG Conversion
If your input images are in HEIC format, convert them to PNG using the `heic2png.py` script.

#### Web UI
Start the local browser UI from the repository root:

```bash
python app.py
```

Open `http://127.0.0.1:5000`, select at least two JPG room photos, and follow the upload, point-cloud conversion, and registration steps. After registration, inspect the merged point cloud in the interactive home-page preview: drag to orbit and scroll to zoom. The merged point cloud and extracted floor-plan point cloud are also available to download. The UI requires the dependencies in `requirements.txt`, the camera calibration file, and the indoor model checkpoint.

Conversion and registration run as background jobs so long-running model
inference does not hold an HTTP request open; the browser polls for completion.

For the Gradio interface, run `python gradio_app.py` and open the local URL it prints. Both interfaces require the model checkpoints and camera calibration file.

#### Render deployment
The root `render.yaml` defines a Docker web service for the Flask UI. In Render,
create a Blueprint from this repository and provide `DEPTH_MODEL_REPO` (the
Hugging Face model repository containing both checkpoints) and `HF_TOKEN` (a
read token for that repository) when prompted. The Docker image and app disable
Xet downloads and use the standard Hub download path. The service uses Render's
`2c-8g` plan; model inference needs more memory than the free plan provides.

#### Hugging Face Gradio Space
The repository is configured as a Gradio Space and supports ZeroGPU for depth
inference. Select ZeroGPU in the Space's hardware settings; the `@spaces.GPU`
wrapper reserves it while the scan callback runs, with a 400-second requested
duration to leave headroom for ZeroGPU's hardware-adjusted request limit. Its
`requirements.txt` is self-contained because the Space build only makes that
file available while installing dependencies. The Dockerfile uses
`requirements-space.txt`. The Space entry point is `gradio_app.py`, and ZeroGPU
requires Python 3.12.12 here. To deploy:

1. Create a **Gradio Space** on Hugging Face. This model requires substantial RAM and CPU Basic may run out of memory; use hardware with enough memory or a GPU for practical inference. Check current hardware pricing before selecting a paid option.
2. In a terminal at the project root, install and authenticate the Hugging Face CLI, then upload the source. Replace `<username>` and `<space-name>` with your Hugging Face account and new Space name:
   ```powershell
   python -m pip install --upgrade huggingface_hub
   hf auth login
   hf repos create <username>/<space-name> --type space --sdk gradio
   hf upload <username>/<space-name> . --type space
   ```
   The project's `.gitignore` excludes local scans, model files, and generated outputs from the source upload. See Hugging Face's [Gradio Spaces guide](https://huggingface.co/docs/hub/spaces-sdks-gradio).
3. If the model weights' license and sharing terms allow, create a separate **private** Hugging Face model repository and upload both required 1.34 GB checkpoints:
   ```powershell
   hf repos create <username>/<private-model-repo> --type model --private
   hf upload <username>/<private-model-repo> checkpoints/depth_anything_metric_depth_indoor.pt depth_anything_metric_depth_indoor.pt --type model
   hf upload <username>/<private-model-repo> checkpoints/depth_anything_vitl14.pth depth_anything_vitl14.pth --type model
   ```
   Do not commit the checkpoint to this app repository.
4. In the Space's Settings, add `DEPTH_MODEL_REPO` as a variable containing the private model repository ID, and `HF_TOKEN` as a secret with read access to that model repository. The app downloads both checkpoints on the first scan. For separate repositories or local encoder checkpoint paths, use `DEPTH_ENCODER_MODEL_REPO`, `DEPTH_ENCODER_FILENAME`, and `DEPTH_ENCODER_CHECKPOINT_PATH`.
5. Confirm `CalibrationMatrix_college_cpt.npz` is present in the Space repo and wait for the build to finish.

The public app has no user login: anyone who can access it can submit scans and use its CPU. Uploaded images and job results are written to the Space's temporary filesystem and may disappear when the Space restarts. Do not use it for sensitive images. For private scans or unrestricted use, use a private Space and add access control before sharing it.

### Model Checkpoints

------------

Model checkpoints necessary for depth estimation can be downloaded from the following Google Drive link:

[Google Drive - Model Checkpoints](https://drive.google.com/drive/folders/1LJRnpOhNuzZXlVE0oGzzUb7ZiXeF6f-8?usp=sharing "Google Drive - Model Checkpoints")

Download and place these checkpoints in the appropriate directory within your project to ensure the model functions correctly.

### Acknowledgements

------------

Special thanks to the developers of the **Depth Anything** model and all contributors who made this project possible. Your work in computer vision and deep learning is greatly appreciated.
