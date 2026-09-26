"""optimize_core 拷贝层测试。运行：
cd <便携版根目录> && PYTHONUTF8=1 ./python_embeded/python.exe ComfyUI/custom_nodes/H3PromptOpt/tests/test_core.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.dirname(HERE)
COMFY = os.path.dirname(os.path.dirname(PLUGIN))
sys.path.insert(0, COMFY)
sys.path.insert(0, PLUGIN)

import optimize_core as oc


def check(name, cond):
    if not cond:
        raise AssertionError("FAIL: " + name)
    print("PASS:", name)


MEDIA = [
    {"label": "picture 1", "kind": "image"},
    {"label": "picture 2", "kind": "image"},
    {"label": "video 1", "kind": "video"},
    {"label": "audio 1", "kind": "audio"},
    {"label": "audio 2", "kind": "audio"},
]

# 1) 标签过滤（对齐 H3MediaBoard test_optimizer.py 已验证用例）
p = "<Picture 1> and <picture 2>, <VIDEO 1> <Audio 2> <picture 1>"
f = oc._filter_media_by_prompt(p, MEDIA)
check("按标签过滤引用", [m["label"] for m in f] == ["picture 1", "picture 2", "video 1", "audio 2"])
check("空提示词过滤为空", oc._filter_media_by_prompt("", MEDIA) == [])
check("无标签过滤为空", oc._filter_media_by_prompt("hello world", MEDIA) == [])
# 2) D5：引用但未提供的素材 → 忽略，不报错
check("引用缺失素材被忽略", oc._filter_media_by_prompt("<Picture 9> x", MEDIA) == [])
check("缺失音视频标签被忽略", oc._filter_media_by_prompt("<Video 3> <Audio 3>", MEDIA) == [])
# 3) label 序号解析
check("label_num 解析", oc._label_num({"label": "picture 12"}) == 12 and oc._label_num({"label": "video 3"}) == 3 and oc._label_num({"label": "x"}) == -1)
# 4) 任务判定
check("T2VA 判定", oc._auto_task("hello world", MEDIA) == "T2VA")
check("Hybrid 判定", oc._auto_task(p, MEDIA) == "HYBRID")
# 5) 系统提示词组装（对齐 H3MediaBoard 已验证用例）
sp3 = oc._system_prompt("T2VA", 5, [], output_language="中文")
sp6 = oc._system_prompt("HYBRID", 5, ["picture 1"], output_language="中文")
check("T2VA 三段式", "integrated_multimodal_description" in sp3 and "subject_definitions" not in sp3)
check("Hybrid 六段式", "subject_definitions" in sp6 and "retention_analysis" in sp6)
check("reference_only 注入", "reference_only" in oc._system_prompt("HYBRID", 5, [], output_language="中文", context={"audio_mode": "reference_only"}))
check("FL2VA 小写标签规则", "picture 1" in oc._system_prompt("FL2VA", 5, [], output_language="中文"))
# 护栏（9B 小模型实测违例后强化）：文本内容不得挂媒体来源标签；retention 只写已供标签
sp_guard = oc._system_prompt("HYBRID", 5, ["picture 1"], output_language="中文")
check("来源引用护栏", "content that appears only in the user's text must not cite any media label" in sp_guard)
check("retention 只写已供标签", "write one compact line per supplied label and never for any other label" in sp_guard)
# 6) 后处理
stripped = oc._strip_filenames("一只 cat.png 和 dog.mp4")
check("strip_filenames 去文件名", "cat.png" not in stripped and "dog.mp4" not in stripped)
sec = oc._format_prompt_sections("subject_definitions: ABC summary: xyz")
check("format_sections 标题独占一行", "subject_definitions:\n" in sec and "\n\nsummary:\n" in sec)
check("normalize 主体括号", oc._normalize_subject_shorthand("S1 S20 S21 S22") == "(S1) (S20) S21 S22")
dlg = oc._extract_explicit_dialogues('他说："你好"')
check("提取显式对白", dlg == [("你好", "Chinese")])
check("对白补全含 d 标签", "<d>[Chinese]你好</d>" in oc._ensure_supplied_dialogues("integrated_multimodal_description:\n你好", '他说："你好"', "中文"))
fl2 = oc._ensure_fl2va_picture_labels("正文", "FL2VA", 5, "中文")
check("FL2VA 标签补全", fl2.startswith("参考图像与目标视频对齐关系") and "picture 1" in fl2)
check("非 FL2VA 不动", oc._ensure_fl2va_picture_labels("正文", "T2VA", 5, "中文") == "正文")
# 7) retention 行清洗（H3PromptOpt 自增后处理）
dirty = ("retention_analysis:\n<picture 1>: fully_preserved - a\n"
         "<audio 1>: reference - b\n<picture 1>: attribute_transfer - c\n\noverall_soundscape:\n安静")
clean = oc.clean_retention_analysis(dirty, ["picture 1"])
check("捏造标签行被丢弃", "<audio 1>" not in clean)
check("重复标签去重", clean.count("<picture 1>:") == 1)
check("无已供标签时原样返回", oc.clean_retention_analysis(dirty, []) == dirty)

print("ALL PASS")
