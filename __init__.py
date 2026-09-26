"""H3PromptOpt —— 工作流式本地提示词优化插件（/upload + /prompt 驱动，本地 GGUF 模型）。

三个节点：H3_OptMediaInput（15 个可选文件名控件 → 请求载荷）
→ H3_OptLocalRun（过滤/任务判定/系统提示词/llama_cpp 生成/后处理 → STRING）
→ H3_OptSaveText（落盘 txt + 原样返回）。
"""
from __future__ import annotations

import logging
import os
import sys as _sys
from datetime import datetime

import folder_paths
from comfy_api.latest import io

# llm_local / media_codec / optimize_core 与 __init__ 同目录。
# ComfyUI 的 load_custom_node 用 spec_from_file_location 执行本文件，不会把插件目录加入 sys.path，
# 因此必须自行注入，否则真实运行时 bare import 失败（对齐 H3MediaBoard 同款做法）。
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
if _PLUGIN_DIR not in _sys.path:
    _sys.path.insert(0, _PLUGIN_DIR)

import llm_local
import media_codec
import optimize_core

REQUEST_IO = io.Custom("H3_OPT_REQUEST")

PICTURE_NAMES = [f"picture_{i}" for i in range(1, 10)]   # picture_1..picture_9
VIDEO_NAMES = [f"video_{i}" for i in range(1, 4)]        # video_1..video_3
AUDIO_NAMES = [f"audio_{i}" for i in range(1, 4)]        # audio_1..audio_3
SLOT_NAMES = PICTURE_NAMES + VIDEO_NAMES + AUDIO_NAMES
SLOT_KIND = {**{n: "image" for n in PICTURE_NAMES},
             **{n: "video" for n in VIDEO_NAMES},
             **{n: "audio" for n in AUDIO_NAMES}}
SLOT_LABEL = {n: f"{n.split('_')[0]} {n.split('_')[1]}" for n in SLOT_NAMES}


def _resolve_media_path(value: str) -> str:
    """annotated 相对路径（"name" / "subfolder/name"）解析到 input 目录；绝对路径原样返回。

    get_annotated_filepath 不支持绝对路径（会把绝对路径拼到 input 目录下触发越界校验），
    因此绝对路径必须先短路（对齐 H3MediaBoard _resolve_media_path 做法）。
    """
    if os.path.isabs(value):
        return value
    return folder_paths.get_annotated_filepath(value)


class H3_OptMediaInput(io.ComfyNode):
    """素材输入节点：提示词 + 15 个可选文件名控件 → H3_OPT_REQUEST 载荷。"""

    @classmethod
    def define_schema(cls):
        inputs = [
            io.String.Input("prompt", display_name="提示词（含 <Picture n> 等占位符）", default="", multiline=True),
            io.Float.Input("duration", display_name="目标时长(秒)", default=5.0, min=1, max=60, step=0.5),
            io.Combo.Input("output_language", display_name="输出语言", options=["中文", "English"], default="中文"),
        ]
        kind_cn = {"picture": "图片", "video": "视频", "audio": "音频"}
        for name in SLOT_NAMES:
            inputs.append(io.String.Input(
                name,
                display_name=f"{kind_cn[name.split('_')[0]]} {name.split('_')[1]} 文件名（可空）",
                default="", optional=True,
            ))
        return io.Schema(
            node_id="H3_OptMediaInput",
            display_name="H3｜优化器素材输入(本地模型)",
            category="H3PromptOpt",
            inputs=inputs,
            outputs=[REQUEST_IO.Output(display_name="request")],
        )

    @classmethod
    def execute(cls, prompt="", duration=5.0, output_language="中文", **slots):
        media = []
        for name in SLOT_NAMES:
            value = str(slots.get(name) or "").strip()
            if not value:
                continue
            kind = SLOT_KIND[name]
            try:
                path = _resolve_media_path(value)
                images = media_codec.media_item_images(kind, path)
                item = {"label": SLOT_LABEL[name], "kind": kind, "images": images}
                if kind == "audio":
                    item["duration"] = media_codec.media_duration(path)
                media.append(item)
            except Exception as exc:
                logging.warning("[H3PromptOpt] 跳过无法加载的素材 %s (%s): %s", value, kind, exc)
        return io.NodeOutput({
            "prompt": str(prompt or ""),
            "duration": float(duration),
            "output_language": str(output_language),
            "media": media,
        })


class H3_OptLocalRun(io.ComfyNode):
    """本地模型执行节点：请求载荷 → llama_cpp 生成 → 后处理 → STRING。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3_OptLocalRun",
            display_name="H3｜本地模型提示词优化(27B GGUF)",
            category="H3PromptOpt",
            inputs=[
                REQUEST_IO.Input("request", display_name="请求载荷"),
                io.String.Input("model_file", display_name="模型文件", default=llm_local.MODEL_DEFAULTS["model_file"]),
                io.String.Input("mmproj_file", display_name="投影器文件", default=llm_local.MODEL_DEFAULTS["mmproj_file"]),
                io.Combo.Input("device", display_name="推理设备", options=["cuda", "cpu", "auto"], default=llm_local.MODEL_DEFAULTS["device"]),
                io.Int.Input("max_tokens", display_name="最大生成 token", default=llm_local.MODEL_DEFAULTS["max_tokens"], min=256, max=16384),
                io.Boolean.Input("keep_loaded", display_name="常驻模型(省加载时间)",
                                 default=False,
                                 tooltip="True 时跨请求复用模型实例；27B 常驻显存约 20GB 级，注意与其它工作流并发的 OOM 风险"),
            ],
            outputs=[io.String.Output(display_name="text")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, request=None, model_file="", mmproj_file="", device="cuda", max_tokens=4096, keep_loaded=False):
        request = request if isinstance(request, dict) else {}
        prompt = str(request.get("prompt") or "").strip()
        if not prompt:
            raise RuntimeError("提示词为空：请在 H3_OptMediaInput 的 prompt 控件填写提示词")
        media = request.get("media") if isinstance(request.get("media"), list) else []
        duration = float(request.get("duration") or 5)
        output_language = str(request.get("output_language") or "中文")

        referenced = optimize_core._filter_media_by_prompt(prompt, media)  # 引用缺失的素材自然被忽略（D5）
        task = optimize_core._auto_task(prompt, referenced)
        context = {}
        if any(str(item.get("kind") or "").lower() == "audio" for item in referenced):
            context["audio_mode"] = "reference_only"
        labels = [str(item.get("label") or "") for item in referenced if item.get("label")]
        system = optimize_core._system_prompt(task, duration, labels, output_language, context, prompt)

        config = {
            "model_file": model_file or llm_local.MODEL_DEFAULTS["model_file"],
            "mmproj_file": mmproj_file or llm_local.MODEL_DEFAULTS["mmproj_file"],
            "device": str(device),
            "max_tokens": int(max_tokens),
            "keep_loaded": bool(keep_loaded),
        }
        result = llm_local.generate(config, {"prompt": prompt, "media": referenced, "system": system})
        if not result:
            raise RuntimeError("本地模型返回了空提示词")

        formatted = optimize_core._format_prompt_sections(optimize_core._strip_filenames(result))
        formatted = optimize_core.clean_retention_analysis(formatted, labels)
        formatted = optimize_core._ensure_fl2va_picture_labels(formatted, task, duration, output_language)
        formatted = optimize_core._ensure_supplied_dialogues(formatted, prompt, output_language)
        formatted = optimize_core._normalize_subject_shorthand(formatted)
        return io.NodeOutput(formatted, ui={"text": (formatted,)})


class H3_OptSaveText(io.ComfyNode):
    """字符串输出节点：写 output/h3promptopt/*.txt 并原样返回 STRING。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3_OptSaveText",
            display_name="H3｜优化结果落盘输出(txt)",
            category="H3PromptOpt",
            inputs=[io.String.Input("text", display_name="文本", default="", multiline=True)],
            outputs=[io.String.Output(display_name="text")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, text=""):
        text = str(text or "")
        try:
            out_dir = os.path.join(folder_paths.get_output_directory(), "h3promptopt")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt")
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(text)
        except Exception as exc:
            logging.warning("[H3PromptOpt] 结果落盘失败（不影响返回）: %s", exc)
        return io.NodeOutput(text, ui={"text": (text,)})


NODE_CLASS_MAPPINGS = {
    "H3_OptMediaInput": H3_OptMediaInput,
    "H3_OptLocalRun": H3_OptLocalRun,
    "H3_OptSaveText": H3_OptSaveText,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3_OptMediaInput": "H3｜优化器素材输入(本地模型)",
    "H3_OptLocalRun": "H3｜本地模型提示词优化(27B GGUF)",
    "H3_OptSaveText": "H3｜优化结果落盘输出(txt)",
}
