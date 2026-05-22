import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import gradio as gr
import insightface
import numpy as np
import torch
import yaml
from PIL import Image

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.transforms import get_transforms
from src.models.detector import LiteCueNet

try:
    from huggingface_hub import hf_hub_download
except Exception:  # noqa: BLE001
    hf_hub_download = None


def load_checkpoint_safely(checkpoint_path: str, device: torch.device):
    """兼容不同 torch 版本，并尽量使用更安全的加载方式。"""
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        # 旧版本 torch 不支持 weights_only 参数
        return torch.load(checkpoint_path, map_location=device)


def normalize_video_input(video_file) -> str:
    """
    兼容不同 gradio 版本的 Video 输入:
    - str: 直接是文件路径
    - dict: 可能包含 path/name/video 键
    """
    if video_file is None:
        return ""
    if isinstance(video_file, str):
        return video_file
    if isinstance(video_file, dict):
        for key in ("path", "name", "video"):
            value = video_file.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def normalize_image_input(image_file) -> str:
    """
    兼容不同 gradio 版本的 Image 输入:
    - str: 直接是文件路径
    - dict: 可能包含 path/name/image 键
    """
    if image_file is None:
        return ""
    if isinstance(image_file, str):
        return image_file
    if isinstance(image_file, dict):
        for key in ("path", "name", "image"):
            value = image_file.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def load_checkpoint_compat(checkpoint_path: str, device: torch.device):
    try:
        return torch.load(checkpoint_path, map_location=device)
    except Exception:  # noqa: BLE001
        return torch.load(checkpoint_path, map_location=device, weights_only=False)


def download_with_fallback(repo_id: str, candidates: List[str], local_dir: str) -> str:
    if hf_hub_download is None:
        raise RuntimeError("未安装 huggingface_hub，请先安装后再使用 FSFM。")
    last_error = None
    for filename in candidates:
        try:
            return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
        except Exception as e:  # noqa: BLE001
            last_error = e
    raise RuntimeError(f"无法下载 FSFM 权重，最后错误: {last_error}")


class FSFMImageDetector:
    HF_REPO_ID = "Wolowolo/fsfm-3c"
    DEFAULT_CKPT_CANDIDATES = [
        "finetuned_models/FS-VFM_extensions/cross_dataset_DFD_and_DiFF/ViT-B_VF2_600e/FT_on_FF++_c23_32frames/checkpoint-min_val_loss.pth",
        "finetuned_models/FS-VFM_extensions/cross_dataset_DFD_and_DiFF/ViT-B_VF2_600e/FT_on_FF++_DF_c23_32frames/checkpoint-min_val_loss.pth",
    ]
    DEFAULT_MEAN_STD_CANDIDATES = [
        "finetuned_models/FS-VFM_extensions/cross_dataset_DFD_and_DiFF/ViT-B_VF2_600e/FT_on_FF++_c23_32frames/pretrain_ds_mean_std.txt",
        "finetuned_models/FS-VFM_extensions/cross_dataset_DFD_and_DiFF/ViT-B_VF2_600e/FT_on_FF++_DF_c23_32frames/pretrain_ds_mean_std.txt",
    ]

    def __init__(self, device: torch.device, model_name: str = "vit_base_patch16"):
        self.device = device
        self.model_name = model_name
        fsfm_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../FSFM/FSFM-CVPR25/fsvfm"))
        if fsfm_root not in sys.path:
            sys.path.insert(0, fsfm_root)
        try:
            import models_vit  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"无法导入 FSFM 模型代码: {e}")

        weights_dir = os.path.abspath(os.path.join("runs", "fsfm_weights"))
        os.makedirs(weights_dir, exist_ok=True)
        ckpt_path = download_with_fallback(self.HF_REPO_ID, self.DEFAULT_CKPT_CANDIDATES, weights_dir)
        mean_std_path = download_with_fallback(self.HF_REPO_ID, self.DEFAULT_MEAN_STD_CANDIDATES, weights_dir)

        with open(mean_std_path, "r", encoding="utf-8") as f:
            stats = json.loads(f.readline().strip())
        mean, std = stats["mean"], stats["std"]

        self.transform = torch.nn.Sequential()  # placeholder to keep attribute existence clear
        self._mean = torch.tensor(mean).view(3, 1, 1)
        self._std = torch.tensor(std).view(3, 1, 1)

        self.model = models_vit.__dict__[model_name](num_classes=2, drop_path_rate=0.1, global_pool=True)
        ckpt = load_checkpoint_compat(ckpt_path, device=torch.device("cpu"))
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, face_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).resize((224, 224), Image.BICUBIC)
        arr = np.asarray(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        tensor = (tensor - self._mean) / self._std
        return tensor.unsqueeze(0)

    def predict_face(self, face_bgr: np.ndarray) -> Tuple[float, float]:
        x = self.preprocess(face_bgr).to(self.device)
        with torch.no_grad():
            # timm 新旧版本对 forward(attn_mask=...) 的签名存在差异；
            # 这里显式走 forward_features + head，避免调用不兼容的 forward。
            feats = self.model.forward_features(x)
            # FSFM 的 models_vit 在 global_pool=True 时 forward_features 已返回 [B, C] 特征，
            # 直接接分类头最稳妥，避免不同 timm 版本的 forward_head pool 参数差异。
            if hasattr(self.model, "head"):
                logits = self.model.head(feats)
            else:
                logits = feats
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().tolist()
        return float(probs[0]), float(probs[1])


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_yaml_path = config.get("dataset_config")
    if dataset_yaml_path:
        with open(dataset_yaml_path, "r", encoding="utf-8") as df:
            dataset_config = yaml.safe_load(df)
        config.update(dataset_config)
    return config


def resolve_checkpoint_path(ckpt_path: str) -> str:
    if os.path.isfile(ckpt_path):
        return ckpt_path

    if not os.path.isdir(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    possible_names = [
        "best_model.pth.tar",
        "model_best.pth.tar",
        "latest_checkpoint.pth.tar",
        "best_model.pth",
        "model_best.pth",
    ]
    for name in possible_names:
        p = os.path.join(ckpt_path, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"No checkpoint file found in: {ckpt_path}")


def load_model(config: Dict, checkpoint_path: str, device: torch.device) -> LiteCueNet:
    model_cfg = config["model"]
    model = LiteCueNet(
        feature_dim=model_cfg["feature_dim"],
        clip_num=model_cfg["clip_num"],
        clip_len=model_cfg["clip_len"],
        num_classes=model_cfg["num_classes"],
        backbone_name=model_cfg["backbone"],
        token_dropout=model_cfg.get("token_dropout", 0.0),
        use_temporal_diff=model_cfg.get("use_temporal_diff", False),
        use_frequency_branch=model_cfg.get("use_frequency_branch", False),
        frequency_fuse_block=model_cfg.get("frequency_fuse_block", 2),
        temporal_module=model_cfg.get("temporal_module", "gated_mlp"),
    ).to(device)

    ckpt = load_checkpoint_safely(checkpoint_path, device=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if len(state_dict) > 0 and list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


class DemoFaceDetector:
    def __init__(self, device: str = "cuda", det_size: Tuple[int, int] = (640, 640)):
        providers = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            root="~/.insightface/models",
            providers=providers,
        )
        self.app.prepare(ctx_id=0, det_size=det_size)

    def detect_largest_face(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        faces = self.app.get(frame_bgr)
        if len(faces) == 0:
            return None
        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        h, w = frame_bgr.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(1, min(x2, w))
        y2 = max(1, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2


def sample_frame_indices(total_frames: int, target_count: int, oversample: int = 3) -> List[int]:
    if total_frames <= 0:
        return []
    count = min(total_frames, max(target_count, target_count * oversample))
    if count == 1:
        return [0]
    return np.linspace(0, total_frames - 1, count, dtype=np.int32).tolist()


class GradCAMCapture:
    """捕获指定卷积层的激活和梯度，用于生成 fake 类 Grad-CAM。"""

    def __init__(self, target_layer: torch.nn.Module):
        self.activations = None
        self.gradients = None
        self._forward_handle = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, _module, _inputs, output):
        self.activations = output
        if isinstance(output, torch.Tensor) and output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient):
        self.gradients = gradient

    def close(self):
        self._forward_handle.remove()


def get_gradcam_target_layer(model: LiteCueNet) -> torch.nn.Module:
    return model.backbone.backbone.blocks[-1]


def build_gradcam_maps(activations: torch.Tensor, gradients: torch.Tensor) -> np.ndarray:
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cams = torch.relu((weights * activations).sum(dim=1))
    cams_np = cams.detach().cpu().numpy()
    flat = cams_np.reshape(cams_np.shape[0], -1)
    mins = flat.min(axis=1).reshape(-1, 1, 1)
    maxs = flat.max(axis=1).reshape(-1, 1, 1)
    return (cams_np - mins) / np.maximum(maxs - mins, 1e-6)


def overlay_heatmap_on_face(frame_bgr: np.ndarray, face_box: Tuple[int, int, int, int], cam: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = face_box
    vis = frame_bgr.copy()
    face_w, face_h = x2 - x1, y2 - y1
    if face_w <= 0 or face_h <= 0:
        return vis

    cam_resized = cv2.resize(cam, (face_w, face_h), interpolation=cv2.INTER_LINEAR)
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(vis[y1:y2, x1:x2], 0.55, heatmap, 0.45, 0)
    vis[y1:y2, x1:x2] = blended
    cv2.rectangle(vis, (x1, y1), (x2, y2), (30, 30, 255), 2)
    return vis


def save_keyframe_heatmaps(
    video_path: str,
    sampled: List[Dict],
    cam_maps: np.ndarray,
    clip_fake_probs: np.ndarray,
    clip_num: int,
    clip_len: int,
    output_dir: str,
    max_keyframes: int = 6,
) -> List[str]:
    keyframe_paths = []
    ranked_clips = np.argsort(-clip_fake_probs)[: min(max_keyframes, clip_num)]
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    try:
        for rank, clip_idx in enumerate(ranked_clips, start=1):
            start = int(clip_idx) * clip_len
            end = min(start + clip_len, len(sampled), len(cam_maps))
            if start >= end:
                continue

            clip_cam_scores = cam_maps[start:end].reshape(end - start, -1).mean(axis=1)
            frame_offset = int(np.argmax(clip_cam_scores))
            sample_idx = start + frame_offset
            sample = sampled[sample_idx]

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(sample["frame_idx"]))
            ok, frame = cap.read()
            if not ok:
                continue

            vis = overlay_heatmap_on_face(frame, sample["face_box"], cam_maps[sample_idx])
            cv2.putText(
                vis,
                f"Keyframe {rank} | Clip:{int(clip_idx)} Fake:{clip_fake_probs[int(clip_idx)]:.3f}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (30, 30, 255),
                2,
            )

            out_path = os.path.abspath(
                os.path.join(
                    output_dir,
                    f"{video_name}_heatmap_rank{rank}_frame{int(sample['frame_idx'])}.jpg",
                )
            )
            cv2.imwrite(out_path, vis)
            keyframe_paths.append(out_path)
    finally:
        cap.release()

    return keyframe_paths


def ensure_web_playable_video(src_path: str) -> str:
    """
    尝试转码为浏览器通用的 H264/yuv420p。
    如果本机无 ffmpeg 或转码失败，则回退原视频。
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        return src_path

    dst_path = src_path.replace(".mp4", "_h264.mp4")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        src_path,
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        dst_path,
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode == 0 and os.path.exists(dst_path):
            return dst_path
    except Exception:
        pass
    return src_path


def infer_and_localize(
    video_path: str,
    model: LiteCueNet,
    detector: DemoFaceDetector,
    transform,
    clip_num: int,
    clip_len: int,
    device: torch.device,
    fake_threshold: float = 0.5,
    max_keyframes: int = 6,
) -> Tuple[Dict, str, List[str]]:
    required_frames = clip_num * clip_len
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    candidate_indices = sample_frame_indices(total_frames, required_frames)
    sampled = []
    cap = cv2.VideoCapture(video_path)
    for idx in candidate_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        face_box = detector.detect_largest_face(frame)
        if face_box is None:
            continue
        x1, y1, x2, y2 = face_box
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue

        resized_roi = cv2.resize(face_roi, (224, 224), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2RGB)
        tensor = transform(Image.fromarray(rgb))
        sampled.append(
            {
                "frame_idx": int(idx),
                "face_box": (x1, y1, x2, y2),
                "roi_224_bgr": resized_roi,
                "tensor": tensor,
            }
        )
        if len(sampled) >= required_frames:
            break
    cap.release()

    if len(sampled) == 0:
        raise RuntimeError("未检测到有效人脸，无法推理。")

    while len(sampled) < required_frames:
        sampled.append(sampled[-1])

    input_tensor = torch.stack([x["tensor"] for x in sampled], dim=0)
    input_tensor = input_tensor.view(clip_num, clip_len, *input_tensor.shape[1:])
    input_tensor = input_tensor.unsqueeze(0).to(device)

    gradcam = GradCAMCapture(get_gradcam_target_layer(model))
    try:
        model.zero_grad(set_to_none=True)
        video_logits, clip_logits = model(input_tensor)
        probs = torch.softmax(video_logits, dim=1)[0].detach().cpu().numpy().tolist()
        clip_fake_probs = torch.softmax(clip_logits, dim=2)[0, :, 1].detach().cpu().numpy()
        video_logits[:, 1].sum().backward()
        if gradcam.activations is None or gradcam.gradients is None:
            raise RuntimeError("Grad-CAM 捕获失败：未获得目标层激活或梯度。")
        cam_maps = build_gradcam_maps(gradcam.activations, gradcam.gradients)
    finally:
        gradcam.close()
        model.zero_grad(set_to_none=True)

    real_prob, fake_prob = float(probs[0]), float(probs[1])
    pred_label = 1 if fake_prob >= fake_threshold else 0

    os.makedirs("runs/demo_outputs", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.abspath(os.path.join("runs/demo_outputs", f"demo_{stamp}"))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.abspath(os.path.join(output_dir, f"demo_{stamp}.mp4"))
    keyframe_paths = []
    if pred_label == 1:
        keyframe_paths = save_keyframe_heatmaps(
            video_path=video_path,
            sampled=sampled,
            cam_maps=cam_maps,
            clip_fake_probs=clip_fake_probs,
            clip_num=clip_num,
            clip_len=clip_len,
            output_dir=output_dir,
            max_keyframes=max_keyframes,
        )

    reader = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    i = 0
    boxed_frames = 0
    while True:
        ok, frame = reader.read()
        if not ok:
            break

        # 演示简化策略：
        # 1) 预测为真实：不画任何框
        # 2) 预测为伪造：直接框出该帧最大人脸
        if pred_label == 1:
            face_box = detector.detect_largest_face(frame)
            if face_box is not None:
                x1, y1, x2, y2 = face_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 30, 255), 2)
                cv2.putText(frame, "Fake Face", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 255), 2)
                boxed_frames += 1

        label_text = f"Real:{real_prob:.3f} Fake:{fake_prob:.3f}"
        verdict = "FAKE" if pred_label == 1 else "REAL"
        color = (30, 30, 255) if pred_label == 1 else (30, 180, 30)
        cv2.putText(frame, f"{verdict} | {label_text}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        writer.write(frame)
        i += 1

    reader.release()
    writer.release()
    playable_output_path = ensure_web_playable_video(output_path)

    result = {
        "video": os.path.basename(video_path),
        "prediction": "fake" if pred_label == 1 else "real",
        "probability": {
            "real": round(real_prob, 6),
            "fake": round(fake_prob, 6),
        },
        "sampled_frames_used": len(sampled),
        "total_frames": total_frames,
        "localized_frames": boxed_frames,
        "heatmap_keyframes": len(keyframe_paths),
    }
    return result, playable_output_path, keyframe_paths


def infer_and_localize_image(
    image_path: str,
    model: LiteCueNet,
    detector: DemoFaceDetector,
    transform,
    clip_num: int,
    clip_len: int,
    device: torch.device,
    fake_threshold: float = 0.5,
) -> Tuple[Dict, str]:
    frame = cv2.imread(image_path)
    if frame is None:
        raise RuntimeError(f"无法读取图片: {image_path}")

    face_box = detector.detect_largest_face(frame)
    if face_box is None:
        raise RuntimeError("图片中未检测到有效人脸，无法推理。")

    x1, y1, x2, y2 = face_box
    face_roi = frame[y1:y2, x1:x2]
    if face_roi.size == 0:
        raise RuntimeError("检测到的人脸区域为空，无法推理。")

    resized_roi = cv2.resize(face_roi, (224, 224), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2RGB)
    tensor = transform(Image.fromarray(rgb))

    required_frames = clip_num * clip_len
    input_tensor = torch.stack([tensor] * required_frames, dim=0)
    input_tensor = input_tensor.view(clip_num, clip_len, *input_tensor.shape[1:])
    input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        video_logits, _ = model(input_tensor)
        probs = torch.softmax(video_logits, dim=1)[0].cpu().numpy().tolist()

    real_prob, fake_prob = float(probs[0]), float(probs[1])
    pred_label = 1 if fake_prob >= fake_threshold else 0

    vis = frame.copy()
    if pred_label == 1:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (30, 30, 255), 2)
        cv2.putText(vis, "Fake Face", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 255), 2)
    else:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (30, 180, 30), 2)
        cv2.putText(vis, "Real Face", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 180, 30), 2)

    verdict = "FAKE" if pred_label == 1 else "REAL"
    color = (30, 30, 255) if pred_label == 1 else (30, 180, 30)
    cv2.putText(
        vis,
        f"{verdict} | Real:{real_prob:.3f} Fake:{fake_prob:.3f}",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )

    os.makedirs("runs/demo_outputs", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.abspath(os.path.join("runs/demo_outputs", f"{image_name}_{stamp}_result.jpg"))
    cv2.imwrite(output_path, vis)

    result = {
        "image": os.path.basename(image_path),
        "prediction": "fake" if pred_label == 1 else "real",
        "probability": {
            "real": round(real_prob, 6),
            "fake": round(fake_prob, 6),
        },
        "face_box": [int(x1), int(y1), int(x2), int(y2)],
    }
    return result, output_path


def infer_and_localize_image_fsfm(
    image_path: str,
    detector: DemoFaceDetector,
    fake_threshold: float = 0.5,
    fsfm_env_name: str = "fsfm-infer",
) -> Tuple[Dict, str]:
    fsfm_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../FSFM/infer_fsvfm.py"))
    if not os.path.isfile(fsfm_script):
        raise RuntimeError(f"未找到 FSFM 推理脚本: {fsfm_script}")

    cmd = [
        "conda",
        "run",
        "-n",
        fsfm_env_name,
        "python",
        fsfm_script,
        "--image",
        image_path,
        "--device",
        "auto",
        "--json-only",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"FSFM 子进程推理失败（环境: {fsfm_env_name}）: {err}")

    fsfm_result = None
    for line in reversed((proc.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                fsfm_result = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if fsfm_result is None:
        raise RuntimeError(f"FSFM 输出解析失败，原始输出: {(proc.stdout or '').strip()}")

    frame = cv2.imread(image_path)
    if frame is None:
        raise RuntimeError(f"无法读取图片: {image_path}")
    face_box = detector.detect_largest_face(frame)
    if face_box is None:
        raise RuntimeError("图片中未检测到有效人脸，无法推理。")

    x1, y1, x2, y2 = face_box
    face_roi = frame[y1:y2, x1:x2]
    if face_roi.size == 0:
        raise RuntimeError("检测到的人脸区域为空，无法推理。")

    real_prob = float(fsfm_result.get("real_prob", 0.0))
    fake_prob = float(fsfm_result.get("fake_prob", 0.0))
    pred_label = 1 if fake_prob >= fake_threshold else 0

    vis = frame.copy()
    color = (30, 30, 255) if pred_label == 1 else (30, 180, 30)
    tag = "Fake Face" if pred_label == 1 else "Real Face"
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    cv2.putText(vis, tag, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    verdict = "FAKE" if pred_label == 1 else "REAL"
    cv2.putText(vis, f"{verdict} | Real:{real_prob:.3f} Fake:{fake_prob:.3f}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    os.makedirs("runs/demo_outputs", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.abspath(os.path.join("runs/demo_outputs", f"{image_name}_{stamp}_fsfm.jpg"))
    cv2.imwrite(output_path, vis)

    result = {
        "image": os.path.basename(image_path),
        "prediction": "fake" if pred_label == 1 else "real",
        "probability": {"real": round(real_prob, 6), "fake": round(fake_prob, 6)},
        "face_box": [int(x1), int(y1), int(x2), int(y2)],
    }
    return result, output_path


def build_demo(config_path: str, checkpoint_path: str, host: str, port: int):
    config = load_config(config_path)
    model_cfg = config["model"]
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    ckpt = resolve_checkpoint_path(checkpoint_path)

    print(f"[Demo] Device: {device}")
    print(f"[Demo] Config: {config_path}")
    print(f"[Demo] Checkpoint: {ckpt}")

    model = load_model(config, ckpt, device=device)
    detector = DemoFaceDetector(device=device_name)
    transform = get_transforms(mode="val")

    def run_demo(video_file, fake_threshold: float):
        video_path = normalize_video_input(video_file)
        if not video_path:
            raise gr.Error("请先拖入或上传一个视频文件。")
        result, out_path, keyframe_paths = infer_and_localize(
            video_path=video_path,
            model=model,
            detector=detector,
            transform=transform,
            clip_num=model_cfg["clip_num"],
            clip_len=model_cfg["clip_len"],
            device=device,
            fake_threshold=fake_threshold,
        )
        text = (
            f"判定结果: {result['prediction'].upper()}\n"
            f"真实概率: {result['probability']['real']:.4f}\n"
            f"伪造概率: {result['probability']['fake']:.4f}\n"
            f"定位帧数: {result['localized_frames']}\n"
            f"热力图关键帧: {result['heatmap_keyframes']}"
        )
        return text, out_path, out_path, keyframe_paths

    def run_image_demo(image_file, fake_threshold: float, image_model_name: str):
        image_path = normalize_image_input(image_file)
        if not image_path:
            raise gr.Error("请先上传一张图片。")
        if image_model_name == "FSFM (HF预训练)":
            result, out_path = infer_and_localize_image_fsfm(
                image_path=image_path,
                detector=detector,
                fake_threshold=fake_threshold,
            )
        else:
            result, out_path = infer_and_localize_image(
                image_path=image_path,
                model=model,
                detector=detector,
                transform=transform,
                clip_num=model_cfg["clip_num"],
                clip_len=model_cfg["clip_len"],
                device=device,
                fake_threshold=fake_threshold,
            )
        text = (
            f"使用模型: {image_model_name}\n"
            f"判定结果: {result['prediction'].upper()}\n"
            f"真实概率: {result['probability']['real']:.4f}\n"
            f"伪造概率: {result['probability']['fake']:.4f}\n"
            f"人脸框位置: {result['face_box']}"
        )
        return text, out_path, out_path

    video_demo = gr.Interface(
        fn=run_demo,
        inputs=[
            gr.Video(label="拖拽或上传测试视频到这里", sources=["upload"], height=420),
            gr.Slider(minimum=0.1, maximum=0.9, step=0.05, value=0.5, label="伪造判定阈值"),
        ],
        outputs=[
            gr.Textbox(label="推理结果"),
            gr.Video(show_label=False, height=420),
            gr.File(label="输出视频下载（播放器不兼容时请下载）"),
            gr.Gallery(label="伪造关键帧热力图", columns=3, height=420),
        ],
        title="视频鉴伪",
        description="拖入视频后输出真假判断与概率；若判定为伪造，将生成 Grad-CAM 关键帧热力图用于提示模型关注区域。",
        flagging_mode="never",
    )

    with gr.Blocks(title="图片鉴伪") as image_demo:
        gr.Markdown("### 图片鉴伪\n上传单张图片后检测最大人脸并输出真假判断与概率。")
        with gr.Row():
            image_in = gr.Image(label="输入图片", type="filepath", height=420)
            image_out = gr.Image(label="输出图片（标注可疑区域）", type="filepath", height=420)
        with gr.Row():
            image_model_name = gr.Radio(
                choices=["LiteCue-Net (本地权重)", "FSFM (HF预训练)"],
                value="LiteCue-Net (本地权重)",
                label="图片鉴伪模型",
            )
            fake_threshold = gr.Slider(minimum=0.1, maximum=0.9, step=0.05, value=0.5, label="伪造判定阈值")
        run_btn = gr.Button("开始鉴伪", variant="primary")
        result_text = gr.Textbox(label="推理结果")
        result_file = gr.File(label="输出图片下载")
        run_btn.click(
            fn=run_image_demo,
            inputs=[image_in, fake_threshold, image_model_name],
            outputs=[result_text, image_out, result_file],
        )

    demo = gr.TabbedInterface(
        [video_demo, image_demo],
        tab_names=["视频鉴伪", "图片鉴伪"],
        title="LiteCue-Net 真伪演示",
    )
    demo.launch(server_name=host, server_port=port)


def parse_args():
    parser = argparse.ArgumentParser(description="LiteCue-Net Gradio Demo")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="训练配置路径")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/exp_20260511/best_model.pth", help="权重文件或权重目录")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Gradio 监听地址")
    parser.add_argument("--port", type=int, default=7860, help="Gradio 端口")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_demo(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        host=args.host,
        port=args.port,
    )
