"""H3PromptOpt 本地 GGUF 视觉模型推理（llama_cpp）——最小实现，逻辑对齐 H3MediaBoard _gguf_generate。

生成前后释放 ComfyUI 管理的模型（unload_all_models + soft_empty_cache）；
keep_loaded=False（默认）每次运行加载/卸载，True 时跨请求复用模块级缓存实例。
"""
from __future__ import annotations

import gc
import re
import threading
from pathlib import Path

# ============================================================
# ★ 模型配置 —— 需要换模型时只改这里（无需翻找其他代码）
#   路径规则：相对路径按 ComfyUI/models/llm 解析；绝对路径原样使用
#   handler：文件名含 qwen3.5/3.6/3.8 → Qwen35ChatHandler；
#            含 qwen3-vl → Qwen3VLChatHandler；其他暂不支持
# ============================================================
MODEL_DEFAULTS = {
    "model_file":  "Qwen3.8-27B-UD-Q2_K_XL.gguf",  # 视觉主模型（27B，Q2_K_XL 量化）
    "mmproj_file": "mmproj-Qwen3.8-27B-F16.gguf",  # 视觉投影器（mmproj）
    "device":      "cuda",                          # cuda / cpu / auto
    "max_tokens":  4096,                            # 生成上限
    "n_ctx":       16384,                           # 上下文窗口
    "temperature": 0.2,
    "top_p":       0.9,
}

_LLM = None          # keep_loaded 缓存（Llama 实例）
_LLM_KEY = None      # 缓存对应的模型配置键
_LLM_LOCK = threading.Lock()


def llm_root() -> Path:
    # 本插件位于 ComfyUI/custom_nodes/H3PromptOpt
    return Path(__file__).resolve().parents[2] / "models" / "llm"


def resolve_model_path(name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else llm_root() / path


def handler_name_for(model_path: str) -> str | None:
    name = Path(model_path).name.lower()
    if re.search(r"qwen[-_. ]?3[._-]?(?:5|6|8)", name):
        return "qwen35"
    if "qwen3-vl" in name or "qwen3vl" in name:
        return "qwen3vl"
    return None


def n_gpu_layers_for(device: str) -> int:
    import torch
    if device == "cpu":
        return 0
    if device == "cuda":
        return -1
    return -1 if torch.cuda.is_available() else 0


def unload_comfy_models() -> None:
    """本地大模型加载前后释放 ComfyUI 管理的模型与缓存（对齐 v3 _unload_comfy_models）。"""
    from comfy import model_management

    model_management.unload_all_models()
    gc.collect()
    model_management.soft_empty_cache()


def user_content(prompt: str, media: list) -> list:
    """构造 llama_cpp chat 的用户内容块（文本 + 每素材每帧 data URL；音频为文字说明）。"""
    content = [{"type": "text", "text": "User prompt:\n" + str(prompt)}]
    for item in media or []:
        label = str(item.get("label") or "")
        kind = str(item.get("kind") or "")
        if kind == "audio":
            duration = item.get("duration")
            suffix = f" ({float(duration):.2f}s)" if isinstance(duration, (int, float)) else ""
            content.append({"type": "text", "text": f"{label} is an uploaded audio reference{suffix}; audio data is not transmitted."})
            continue
        for index, data_url in enumerate(item.get("images") or []):
            content.append({"type": "text", "text": f"{label} visual {index + 1}."})
            content.append({"type": "image_url", "image_url": {"url": data_url}})
    return content


def _model_key(config: dict) -> tuple:
    return (
        str(resolve_model_path(config["model_file"])),
        str(resolve_model_path(config["mmproj_file"])),
        handler_name_for(config["model_file"]),
        str(config.get("device") or "auto"),
    )


def _load(config: dict):
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Qwen35ChatHandler, Qwen3VLChatHandler

    model_path = resolve_model_path(config["model_file"])
    mmproj_path = resolve_model_path(config["mmproj_file"])
    if not model_path.is_file():
        raise ValueError(f"模型文件不存在: {model_path}")
    if not mmproj_path.is_file():
        raise ValueError(f"投影器文件不存在: {mmproj_path}")
    handler_name = handler_name_for(model_path.name)
    if handler_name not in {"qwen35", "qwen3vl"}:
        raise ValueError(f"暂不支持的 GGUF 视觉模型: {model_path.name}（支持 qwen3.5/3.6/3.8 或 qwen3-vl）")
    handler_cls = Qwen35ChatHandler if handler_name == "qwen35" else Qwen3VLChatHandler
    kwargs = {"clip_model_path": str(mmproj_path), "verbose": False}
    if handler_name == "qwen35":
        kwargs["enable_thinking"] = False
    else:
        kwargs["force_reasoning"] = False
    try:
        return Llama(
            model_path=str(model_path),
            chat_handler=handler_cls(**kwargs),
            n_gpu_layers=n_gpu_layers_for(str(config.get("device") or "auto")),
            n_ctx=int(config.get("n_ctx") or MODEL_DEFAULTS["n_ctx"]),
            verbose=False,
        )
    except ValueError as exc:
        if "Failed to load model" in str(exc) or "failed to load" in str(exc).lower():
            raise ValueError(
                f"模型加载失败（可能原因：llama_cpp 版本过旧不支持该模型的张量布局，"
                f"或显卡显存不足）。当前 llama_cpp 要求 >= 0.3.43 的 CUDA 构建"
                f"（本机 whl 目录有 0.3.43/0.3.48+cu130 预编译包）。原始错误: {exc}"
            ) from exc
        raise


def generate(config: dict, payload: dict) -> str:
    """一次本地生成。payload 需含 system / prompt / media 键。"""
    global _LLM, _LLM_KEY
    keep = bool(config.get("keep_loaded"))
    unload_comfy_models()
    try:
        with _LLM_LOCK:
            if _LLM is None or not keep or _LLM_KEY != _model_key(config):
                if _LLM is not None and (not keep or _LLM_KEY != _model_key(config)):
                    # 换模型或 keep=False：先释放旧实例显存再加载新模型，避免新旧双份驻留 OOM
                    _LLM = _LLM_KEY = None
                    gc.collect()
                llm = _load(config)
                if keep:
                    _LLM, _LLM_KEY = llm, _model_key(config)
            else:
                llm = _LLM
            result = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": payload["system"]},
                    {"role": "user", "content": user_content(payload.get("prompt") or "", payload.get("media"))},
                ],
                max_tokens=int(config.get("max_tokens") or 4096),
                temperature=float(config.get("temperature") if config.get("temperature") is not None else MODEL_DEFAULTS["temperature"]),
                top_p=float(config.get("top_p") if config.get("top_p") is not None else MODEL_DEFAULTS["top_p"]),
            )
        return str(result["choices"][0]["message"]["content"] or "").removeprefix(": ").strip()
    finally:
        if not keep:
            with _LLM_LOCK:
                _LLM = _LLM_KEY = None
        unload_comfy_models()
