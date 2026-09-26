"""模板工作流结构测试。运行方式同 test_core.py。"""
import json
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

import H3PromptOpt as plugin


def check(name, cond):
    if not cond:
        raise AssertionError("FAIL: " + name)
    print("PASS:", name)


graph = json.load(open(os.path.join(PLUGIN, "workflows", "prompt_optimize.json"), encoding="utf-8"))
check("三节点齐备", set(graph) >= {"1", "2", "3"})
# 节点 1/2 是本插件节点；节点 3 用核心 SaveText（前端 1.49.6 只为 SaveText/PreviewAny 挂文本预览，
# 自定义节点名没有 UI 预览——用核心节点保证结果在 UI 上可见，/history 契约 outputs["3"]["text"][0] 不变）
check("插件节点已注册", all(graph[node_id]["class_type"] in plugin.NODE_CLASS_MAPPINGS for node_id in ("1", "2")))
check("节点3为核心 SaveText", graph["3"]["class_type"] == "SaveText")
check("链路完整", graph["2"]["inputs"]["request"] == ["1", 0] and graph["3"]["inputs"]["text"] == ["2", 0])
check("SaveText 落盘前缀 h3promptopt", graph["3"]["inputs"].get("filename_prefix") == "h3promptopt" and graph["3"]["inputs"].get("format") == "txt")
inputs1 = graph["1"]["inputs"]
check("15 个素材控件齐全", all(f"{prefix}_{i}" in inputs1 for prefix, count in (("picture", 9), ("video", 3), ("audio", 3)) for i in range(1, count + 1)))
check("必填控件齐全", all(k in inputs1 for k in ("prompt", "duration", "output_language")))
inputs2 = graph["2"]["inputs"]
check("模型控件齐全", all(k in inputs2 for k in ("model_file", "mmproj_file", "device", "max_tokens", "keep_loaded")))

# 8) 输出节点/历史可见性回归（防：/prompt 因无输出节点被 400 拒；/history outputs 缺失）
#    注：本版 RETURN_NAMES 实际为 list（_io.py:2224 赋的是 output_name 列表），故按值等价断言。
check("LocalRun/SaveText 是输出节点(OUTPUT_NODE)",
      plugin.H3_OptLocalRun.OUTPUT_NODE is True and plugin.H3_OptSaveText.OUTPUT_NODE is True)
check("LocalRun 输出名为 text", tuple(plugin.H3_OptLocalRun.RETURN_NAMES) == ("text",))

tmp = tempfile.mkdtemp(prefix="h3opt_wf_")
orig = folder_paths.get_output_directory
folder_paths.get_output_directory = lambda: tmp
try:
    save_out = plugin.H3_OptSaveText.execute(text="hello")
finally:
    folder_paths.get_output_directory = orig
check("SaveText 返回值带 ui.text（/history outputs 可见）",
      save_out.result == ("hello",) and save_out.ui is not None and save_out.ui.get("text") == ("hello",))

print("ALL PASS")
