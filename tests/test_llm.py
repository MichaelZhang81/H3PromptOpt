import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.dirname(HERE)
COMFY = os.path.dirname(os.path.dirname(PLUGIN))
sys.path.insert(0, COMFY)
sys.path.insert(0, PLUGIN)

import llm_local  # noqa: E402


def check(name, cond):
    if not cond:
        raise AssertionError("FAIL: " + name)
    print("PASS:", name)


# 1) 默认配置：键齐全且磁盘上的模型文件真实存在
check("默认配置键齐全", all(k in llm_local.MODEL_DEFAULTS for k in ("model_file", "mmproj_file", "device", "max_tokens", "n_ctx", "temperature", "top_p")))
check("默认模型文件存在", llm_local.resolve_model_path(llm_local.MODEL_DEFAULTS["model_file"]).is_file())
check("默认投影器文件存在", llm_local.resolve_model_path(llm_local.MODEL_DEFAULTS["mmproj_file"]).is_file())

# 2) 路径解析
check("相对路径按 models/llm 解析", str(llm_local.resolve_model_path("x.gguf")).endswith(os.path.join("models", "llm", "x.gguf")))
abs_p = Path(HERE) / "fixtures" / "img.png"
check("绝对路径原样", llm_local.resolve_model_path(str(abs_p)) == abs_p)

# 3) handler 映射
check("qwen3.8 -> qwen35", llm_local.handler_name_for("Qwen3.8-27B-UD-Q2_K_XL.gguf") == "qwen35")
check("qwen3.5/3.6 -> qwen35", llm_local.handler_name_for("Qwen3.5-9B-NSFW-Q8_0.gguf") == "qwen35" and llm_local.handler_name_for("qwen3_6-x.gguf") == "qwen35")
check("qwen3-vl -> qwen3vl", llm_local.handler_name_for("qwen3-vl-8b.gguf") == "qwen3vl")
check("其他 -> None", llm_local.handler_name_for("llama-2-7b.gguf") is None)

# 4) n_gpu_layers
import torch
check("cpu -> 0", llm_local.n_gpu_layers_for("cpu") == 0)
check("cuda -> -1", llm_local.n_gpu_layers_for("cuda") == -1)
check("auto 与 CUDA 可用性一致", llm_local.n_gpu_layers_for("auto") == (-1 if torch.cuda.is_available() else 0))

# 5) user_content 结构（纯函数，不加载模型）
content = llm_local.user_content(
    "hello <Picture 1>",
    [
        {"label": "picture 1", "kind": "image", "images": ["data:image/jpeg;base64,AA"]},
        {"label": "audio 1", "kind": "audio", "duration": 3.2},
    ],
)
check("user_content 文本块开头", content[0] == {"type": "text", "text": "User prompt:\nhello <Picture 1>"})
check("image_url 块成对", content[1] == {"type": "text", "text": "picture 1 visual 1."} and content[2]["type"] == "image_url" and content[2]["image_url"]["url"] == "data:image/jpeg;base64,AA")
check("音频文字说明含时长", content[3]["type"] == "text" and "3.20s" in content[3]["text"] and "audio data is not transmitted" in content[3]["text"])

# 6) _model_key 稳定
check("_model_key 稳定", llm_local._model_key(llm_local.MODEL_DEFAULTS) == llm_local._model_key(llm_local.MODEL_DEFAULTS))

print("ALL PASS")
