"""H3PromptOpt 媒体解码 → 本地模型 data URL。

零新增依赖：av + PIL + InputImpl（与 H3MediaBoard media_loader.py 同源路径）。
视频采样 3 帧（首帧/中间帧/末帧），长边 <=512px，JPEG —— 对齐 H3MediaBoard v3 UI canvas 行为。
音频不解码 PCM（模型听不见），只由 media_duration 提供时长元数据。
"""
from __future__ import annotations

import base64
import io as _io

import av
import torch
from PIL import Image as PILImage
from PIL import ImageOps

from comfy_api.latest import InputImpl

VIDEO_FRAME_COUNT = 3     # 视频采样帧数
VIDEO_MAX_EDGE = 512      # 采样帧长边上限（px）
JPEG_QUALITY = 85


def _to_data_url(rgb: PILImage.Image) -> str:
    buf = _io.BytesIO()
    rgb.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _pil_from_tensor(frame: torch.Tensor) -> PILImage.Image:
    """[H,W,C] float32(0-1) 张量 -> PIL RGB 图。"""
    arr = (frame.detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    return PILImage.fromarray(arr, mode="RGB")


def media_item_images(kind: str, path: str) -> list[str]:
    """按类型解码素材为 data URL 列表：image→1 张原分辨率；video→3 帧；audio→[]。"""
    if kind == "image":
        components = InputImpl.VideoFromFile(path).get_components()
        if components.images is not None and components.images.shape[0] > 0:
            frame = components.images[0]
        else:
            img = ImageOps.exif_transpose(PILImage.open(path)).convert("RGB")
            return [_to_data_url(img)]
        return [_to_data_url(_pil_from_tensor(frame))]
    if kind == "video":
        components = InputImpl.VideoFromFile(path).get_components()
        frames = components.images
        if frames is None or frames.shape[0] == 0:
            raise ValueError("无法解码视频帧: " + path)
        total = frames.shape[0]
        indices = sorted({0, total // 2, total - 1})
        urls = []
        for index in indices:
            img = _pil_from_tensor(frames[index])
            width, height = img.size
            scale = min(1.0, VIDEO_MAX_EDGE / max(width, height))
            if scale < 1.0:
                img = img.resize(
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    PILImage.LANCZOS,
                )
            urls.append(_to_data_url(img))
        return urls
    if kind == "audio":
        return []
    raise ValueError("未知素材类型: " + str(kind))


def media_duration(path: str) -> float:
    """音频文件时长（秒）。无音轨或读不到时长抛 ValueError。"""
    with av.open(path) as af:
        if not af.streams.audio:
            raise ValueError("No audio stream found in the file: " + path)
        stream = af.streams.audio[0]
        if stream.duration is not None:
            duration = float(stream.duration * stream.time_base)
        else:
            duration = float(af.duration / av.time_base)
        if duration <= 0:
            raise ValueError("无法读取音频时长: " + path)
        return duration
