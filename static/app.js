const imageInput = document.querySelector("#image-input");
const dropZone = document.querySelector("#drop-zone");
const uploadButton = document.querySelector("#upload-button");
const convertButton = document.querySelector("#convert-button");
const registerButton = document.querySelector("#register-button");
const selectionRow = document.querySelector("#selection-row");
const selectionCount = document.querySelector("#selection-count");
const selectionNames = document.querySelector("#selection-names");
const notice = document.querySelector("#notice");
const progressArea = document.querySelector("#progress-area");
const results = document.querySelector("#results");
const downloadList = document.querySelector("#download-list");
const cloudCanvas = document.querySelector("#cloud-canvas");
const cloudContext = cloudCanvas.getContext("2d");
const viewerLoading = document.querySelector("#viewer-loading");
const viewerMessage = document.querySelector("#viewer-message");
const viewerStats = document.querySelector("#viewer-stats");

let jobId = null;
let uploaded = false;
let converted = false;
let busy = false;
let cloudData = null;
let cloudView = { yaw: 0.55, pitch: 0.3, zoom: 1 };
let cloudDrag = null;

function syncButtons() {
  const files = Array.from(imageInput.files || []);
  const totalSize = files.reduce((total, file) => total + file.size, 0);
  uploadButton.disabled = busy || files.length < 2 || files.length > 40 || totalSize > 500 * 1024 * 1024;
  convertButton.disabled = busy || !uploaded;
  registerButton.disabled = busy || !converted;
  document.querySelector("#browse-button").disabled = busy;
  document.querySelector("#clear-button").disabled = busy;
}

function showNotice(message, isError = false) {
  notice.textContent = message;
  notice.classList.toggle("error", isError);
  notice.hidden = false;
}

function setBusy(isBusy, title = "", description = "") {
  busy = isBusy;
  progressArea.hidden = !isBusy;
  if (isBusy) {
    document.querySelector("#progress-title").textContent = title;
    document.querySelector("#progress-description").textContent = description;
  }
  syncButtons();
}

function updateSteps(activeStep) {
  document.querySelectorAll(".step").forEach((step) => {
    const number = Number(step.dataset.step);
    step.classList.toggle("active", number === activeStep);
    step.classList.toggle("complete", number < activeStep);
  });
}

function updateSelection() {
  const files = Array.from(imageInput.files || []);
  jobId = null;
  uploaded = false;
  converted = false;
  selectionRow.hidden = files.length === 0;
  selectionCount.textContent = `${files.length} JPG image${files.length === 1 ? "" : "s"} selected`;
  selectionNames.textContent = files.slice(0, 3).map((file) => file.name).join(", ")
    + (files.length > 3 ? ` + ${files.length - 3} more` : "");
  if (files.length === 1) {
    showNotice("Select at least two JPG images to register a room.");
  } else if (files.length > 40) {
    showNotice("Choose no more than 40 images.");
  } else if (files.reduce((total, file) => total + file.size, 0) > 500 * 1024 * 1024) {
    showNotice("The selected images exceed the 500 MB total upload limit.");
  } else {
    notice.hidden = true;
  }
  results.hidden = true;
  cloudData = null;
  updateSteps(1);
  syncButtons();
}

async function requestJson(url, options) {
  const response = await fetch(url, { cache: "no-store", ...options });
  const responseText = await response.text();
  let payload;
  try {
    payload = responseText ? JSON.parse(responseText) : null;
  } catch {
    const detail = responseText.trim().slice(0, 300);
    throw new Error(
      `The server returned a non-JSON response (HTTP ${response.status})${detail ? `: ${detail}` : "."}`,
    );
  }
  if (!payload || typeof payload !== "object") {
    throw new Error(
      `The server returned an empty response (HTTP ${response.status}). ` +
      "The request may have timed out or the app may have restarted. Check the Render service logs.",
    );
  }
  if (!response.ok) {
    throw new Error(payload.error || `The request failed (HTTP ${response.status}).`);
  }
  return payload;
}

async function runPipeline(url) {
  await requestJson(url, { method: "POST" });
  while (true) {
    await new Promise((resolve) => window.setTimeout(resolve, 2000));
    const payload = await requestJson(url);
    if (payload.status !== "queued" && payload.status !== "running") return payload;
  }
}

function drawPointCloud() {
  if (!cloudData) return;
  const bounds = cloudCanvas.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;
  const pixelRatio = window.devicePixelRatio || 1;
  const width = Math.round(bounds.width * pixelRatio);
  const height = Math.round(bounds.height * pixelRatio);
  if (cloudCanvas.width !== width || cloudCanvas.height !== height) {
    cloudCanvas.width = width;
    cloudCanvas.height = height;
  }
  cloudContext.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  cloudContext.fillStyle = "#111a16";
  cloudContext.fillRect(0, 0, bounds.width, bounds.height);

  const { positions, colors, count } = cloudData;
  const cosineYaw = Math.cos(cloudView.yaw);
  const sineYaw = Math.sin(cloudView.yaw);
  const cosinePitch = Math.cos(cloudView.pitch);
  const sinePitch = Math.sin(cloudView.pitch);
  const focalLength = Math.min(bounds.width, bounds.height) * 1.75 * cloudView.zoom;
  const cameraDistance = 3;
  const projected = [];

  for (let index = 0; index < count; index += 1) {
    const offset = index * 3;
    const x = positions[offset];
    const y = positions[offset + 1];
    const z = positions[offset + 2];
    const rotatedX = x * cosineYaw - y * sineYaw;
    const rotatedY = x * sineYaw + y * cosineYaw;
    const cameraY = rotatedY * cosinePitch - z * sinePitch;
    const cameraZ = rotatedY * sinePitch + z * cosinePitch;
    const depth = cameraDistance - cameraZ;
    if (depth <= 0.1) continue;
    projected.push({
      x: bounds.width / 2 + rotatedX * focalLength / depth,
      y: bounds.height / 2 - cameraY * focalLength / depth,
      depth,
      color: `rgb(${colors[offset]}, ${colors[offset + 1]}, ${colors[offset + 2]})`,
      radius: Math.max(0.7, Math.min(1.7, 2.8 / depth)),
    });
  }

  projected.sort((left, right) => right.depth - left.depth);
  for (const point of projected) {
    cloudContext.fillStyle = point.color;
    cloudContext.fillRect(
      point.x,
      point.y,
      point.radius * 2,
      point.radius * 2,
    );
  }
}

async function loadPointCloudPreview() {
  cloudData = null;
  viewerLoading.hidden = false;
  viewerMessage.textContent = "Loading point cloud preview…";
  viewerStats.textContent = "Preparing preview…";
  try {
    const response = await requestJson(`/api/jobs/${jobId}/preview`);
    const positionBytes = Uint8Array.from(atob(response.positions), (character) => character.charCodeAt(0));
    const colorBytes = Uint8Array.from(atob(response.colors), (character) => character.charCodeAt(0));
    cloudData = {
      positions: new Float32Array(positionBytes.buffer),
      colors: colorBytes,
      count: response.count,
    };
    viewerStats.textContent = `${response.count.toLocaleString()} preview points  ·  ${response.dimensions.map((value) => `${value} m`).join(" × ")}`;
    viewerLoading.hidden = true;
    drawPointCloud();
  } catch (error) {
    viewerMessage.textContent = error.message;
    viewerStats.textContent = "Preview unavailable";
  }
}

cloudCanvas.addEventListener("pointerdown", (event) => {
  if (!cloudData) return;
  cloudDrag = { x: event.clientX, y: event.clientY };
  cloudCanvas.setPointerCapture(event.pointerId);
});
cloudCanvas.addEventListener("pointermove", (event) => {
  if (!cloudDrag) return;
  cloudView.yaw += (event.clientX - cloudDrag.x) * 0.008;
  cloudView.pitch = Math.max(-1.35, Math.min(1.35, cloudView.pitch + (event.clientY - cloudDrag.y) * 0.008));
  cloudDrag = { x: event.clientX, y: event.clientY };
  drawPointCloud();
});
cloudCanvas.addEventListener("pointerup", () => { cloudDrag = null; });
cloudCanvas.addEventListener("pointercancel", () => { cloudDrag = null; });
cloudCanvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  cloudView.zoom = Math.max(0.45, Math.min(3.2, cloudView.zoom * Math.exp(-event.deltaY * 0.001)));
  drawPointCloud();
}, { passive: false });
document.querySelector("#viewer-reset").addEventListener("click", () => {
  cloudView = { yaw: 0.55, pitch: 0.3, zoom: 1 };
  drawPointCloud();
});
window.addEventListener("resize", drawPointCloud);

imageInput.addEventListener("change", updateSelection);
document.querySelector("#browse-button").addEventListener("click", () => imageInput.click());
document.querySelector("#clear-button").addEventListener("click", () => {
  imageInput.value = "";
  updateSelection();
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragging"));
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  const files = Array.from(event.dataTransfer.files);
  const invalid = files.find((file) => !/\.jpe?g$/i.test(file.name) || (file.type && file.type !== "image/jpeg"));
  if (invalid) {
    showNotice(`${invalid.name} is not a JPG image.`, true);
    return;
  }
  const transfer = new DataTransfer();
  files.forEach((file) => transfer.items.add(file));
  imageInput.files = transfer.files;
  updateSelection();
});

uploadButton.addEventListener("click", async () => {
  if (imageInput.files.length === 0) return;
  const body = new FormData();
  Array.from(imageInput.files).forEach((file) => body.append("images", file));
  notice.hidden = true;
  setBusy(true, "Uploading your photos", "Checking and organizing the selected JPG images.");
  try {
    const payload = await requestJson("/api/upload", { method: "POST", body });
    jobId = payload.jobId;
    uploaded = true;
    converted = false;
    convertButton.disabled = false;
    registerButton.disabled = true;
    updateSteps(2);
    showNotice(`${payload.images.length} images uploaded and ready to convert.`);
  } catch (error) {
    showNotice(error.message, true);
  } finally {
    setBusy(false);
  }
});

convertButton.addEventListener("click", async () => {
  if (!uploaded || !jobId) return;
  notice.hidden = true;
  setBusy(true, "Building your point clouds", "Estimating metric depth for each image. The first run may take longer.");
  try {
    const payload = await runPipeline(`/api/jobs/${jobId}/convert`);
    converted = true;
    registerButton.disabled = false;
    updateSteps(3);
    showNotice(`${payload.count} point clouds created. You can now register the room scans.`);
  } catch (error) {
    showNotice(error.message, true);
  } finally {
    setBusy(false);
  }
});

registerButton.addEventListener("click", async () => {
  if (!converted || !jobId) return;
  notice.hidden = true;
  results.hidden = true;
  setBusy(true, "Stitching the room together", "Aligning scans and extracting the 2D floor plan.");
  try {
    const payload = await runPipeline(`/api/jobs/${jobId}/register`);
    downloadList.replaceChildren();
    payload.files.forEach((file) => {
      if (file.name === "floor_plan_preview.png") {
        const preview = document.createElement("img");
        preview.className = "floor-plan-preview";
        preview.src = file.url;
        preview.alt = "Cleaned room scan outline with metric dimensions";
        downloadList.append(preview);
        return;
      }
      const link = document.createElement("a");
      link.className = "download-link";
      link.href = file.url;
      link.download = file.name;
      const title = document.createElement("strong");
      title.textContent = file.name === "extracted_floor_plan.ply" ? "Cleaned floor-plan outline" : "Merged room point cloud";
      const detail = document.createElement("span");
      detail.textContent = file.name;
      const arrow = document.createElement("span");
      arrow.className = "download-arrow";
      arrow.textContent = "↓";
      link.append(title, detail, arrow);
      downloadList.append(link);
    });
    results.hidden = false;
    await loadPointCloudPreview();
    updateSteps(4);
  } catch (error) {
    showNotice(error.message, true);
  } finally {
    setBusy(false);
  }
});
