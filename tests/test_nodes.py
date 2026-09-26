"""节点层测试（离线：schema 校验 + execute 管线；模型生成用 monkeypatch 替身）。
运行方式同 test_core.py。
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.dirname(HERE)
COMFY = os.path.dirname(os.path.dirname(PLUGIN))
CUSTOM = os.path.dirname(PLUGIN)
sys.path.insert(0, COMFY)
sys.path.insert(0, CUSTOM)
sys.path.insert(0, PLUGIN)

import folder_paths
import llm_local
from comfy_api.latest import _io

import H3PromptOpt as plugin


def check(name, cond):
    if not cond:
        raise AssertionError("FAIL: " + name)
    print("PASS:", name)


def first(out):
    result = out.result
    return result[0] if isinstance(result, tuple) else result


def run_node(cls, inputs):
    # 本版 comfy_api 签名为 (out_dict, hidden, v3_data) 三元组（_io.py:1822）
    _out_dict, _hidden, v3_data = _io.get_finalized_class_inputs(cls.INPUT_TYPES(), inputs)
    nested = _io.build_nested_inputs(dict(inputs), v3_data)
    return cls.execute(**nested)


FIX = os.path.join(HERE, "fixtures")
IMG = os.path.join(FIX, "img.png")
VID = os.path.join(FIX, "sample.mp4")
AUD = os.path.join(FIX, "tone.wav")

# 1) 注册与 schema
NAMES = ["H3_OptMediaInput", "H3_OptLocalRun", "H3_OptSaveText"]
check("三节点已注册", all(n in plugin.NODE_CLASS_MAPPINGS for n in NAMES))
check("INPUT_TYPES 可序列化", all(isinstance(plugin.NODE_CLASS_MAPPINGS[n].INPUT_TYPES(), dict) for n in NAMES))

# 2) 输入节点：真实解码打包
out1 = run_node(plugin.H3_OptMediaInput, {
    "prompt": "<Picture 1> 一只猫 <Video 1> 奔跑 <Audio 1> 叫声",
    "duration": 5, "output_language": "中文",
    "picture_1": IMG, "video_1": VID, "audio_1": AUD,
})
payload = first(out1)
check("载荷基本字段", payload["prompt"].startswith("<Picture 1>") and payload["duration"] == 5 and payload["output_language"] == "中文")
check("媒体按序打包", [(m["label"], m["kind"]) for m in payload["media"]] == [("picture 1", "image"), ("video 1", "video"), ("audio 1", "audio")])
check("视频 3 帧", len(payload["media"][1]["images"]) == 3)
check("音频含时长且无图像", payload["media"][2].get("duration", 0) > 0 and payload["media"][2]["images"] == [])

# 3) 输入节点：坏路径槽位跳过（不拖垮节点）
out2 = run_node(plugin.H3_OptMediaInput, {"prompt": "x", "duration": 5, "output_language": "中文", "picture_1": "no_such_file.png"})
check("坏路径槽位跳过", first(out2)["media"] == [])

# 4) 模型节点管线（mock 生成；含 9B 违例形态：捏造 <audio 1> 行 + <picture 1> 重复行）
calls = {}
def fake_generate(config, payload):
    calls["config"] = config
    calls["payload"] = payload
    return ("subject_definitions:\n<Subject 1> 源自 <picture 1>，为一只猫。\n\n"
            "summary:\n[reference generation] 基于 <picture 1> 生成视频。\n\n"
            "retention_analysis:\n<picture 1>: fully_preserved - 猫的外形完整保留。\n"
            "<video 1>: attribute_transfer - 动作转移（video 1 未提供，应被丢弃）。\n"
            "<picture 1>: attribute_transfer - 动作转移（重复，应被丢弃）。\n\n"
            "detailed_description:\n一只猫在跑。\n\n"
            "overall_soundscape:\n安静\n\n"
            "non_diegetic_music:\n无")

llm_local.generate = fake_generate
req = {
    # <Picture 9> 故意引用不存在的素材：必须被忽略（D5），不报错
    "prompt": "<Picture 1> 猫 <Audio 1> 叫声 <Picture 9> 不存在", "duration": 5, "output_language": "中文",
    "media": [
        {"label": "picture 1", "kind": "image", "images": ["data:image/jpeg;base64,AA"]},
        {"label": "picture 2", "kind": "image", "images": ["data:image/jpeg;base64,BB"]},
        {"label": "audio 1", "kind": "audio", "duration": 2.5},
    ],
}
out3 = run_node(plugin.H3_OptLocalRun, {"request": req, "model_file": "", "mmproj_file": "", "device": "cuda", "max_tokens": 4096, "keep_loaded": False})
result = first(out3)
check("返回优化字符串", isinstance(result, str) and "一只猫在跑" in result)
check("retention 清洗（捏造 video 1 丢弃、picture 1 去重）", "<video 1>" not in result and result.count("<picture 1>:") == 1)
check("只送被引用素材（picture 2 未引用被滤掉、picture 9 缺失被忽略）", [m["label"] for m in calls["payload"]["media"]] == ["picture 1", "audio 1"])
check("任务判定 HYBRID（六段式系统提示词）", "subject_definitions" in calls["payload"]["system"] and "retention_analysis" in calls["payload"]["system"])
check("音频模式注入", "reference_only" in calls["payload"]["system"])
check("空控件回退默认模型", calls["config"]["model_file"] == llm_local.MODEL_DEFAULTS["model_file"] and calls["config"]["mmproj_file"] == llm_local.MODEL_DEFAULTS["mmproj_file"])

# 5) 空提示词报错
try:
    run_node(plugin.H3_OptLocalRun, {"request": {"prompt": "  ", "media": []}, "model_file": "", "mmproj_file": "", "device": "cuda", "max_tokens": 4096, "keep_loaded": False})
    raise AssertionError("应抛异常")
except RuntimeError:
    print("PASS: 空提示词抛异常")

# 6) 空结果报错
llm_local.generate = lambda config, payload: ""
try:
    run_node(plugin.H3_OptLocalRun, {"request": req, "model_file": "", "mmproj_file": "", "device": "cuda", "max_tokens": 4096, "keep_loaded": False})
    raise AssertionError("应抛异常")
except RuntimeError:
    print("PASS: 空结果抛异常")

# 7) 保存节点：落盘 + 透传
tmp = tempfile.mkdtemp(prefix="h3opt_out_")
orig = folder_paths.get_output_directory
folder_paths.get_output_directory = lambda: tmp
saved = first(run_node(plugin.H3_OptSaveText, {"text": "hello 你好"}))
folder_paths.get_output_directory = orig
files = [f for f in os.listdir(os.path.join(tmp, "h3promptopt")) if f.endswith(".txt")]
check("落盘 txt 且原样返回", saved == "hello 你好" and len(files) == 1)

print("ALL PASS")
