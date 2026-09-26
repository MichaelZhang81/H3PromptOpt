"""media_codec 媒体解码测试。运行方式同 test_core.py。"""
import base64
import io as _io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.dirname(HERE)
COMFY = os.path.dirname(os.path.dirname(PLUGIN))
sys.path.insert(0, COMFY)
sys.path.insert(0, PLUGIN)

import media_codec  # noqa: E402

FIX = os.path.join(HERE, "fixtures")


def check(name, cond):
    if not cond:
        raise AssertionError("FAIL: " + name)
    print("PASS:", name)


def decode(data_url):
    from PIL import Image
    return Image.open(_io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))


u = media_codec.media_item_images("image", os.path.join(FIX, "img.png"))
check("图片出 1 张 data URL", len(u) == 1 and u[0].startswith("data:image/jpeg;base64,"))
check("图片 data URL 可解码", decode(u[0]).size[0] > 0)

v = media_codec.media_item_images("video", os.path.join(FIX, "sample.mp4"))
check("视频出 3 帧", len(v) == 3)
check("视频帧长边 <=512", all(max(decode(x).size) <= 512 for x in v))

check("音频无图像", media_codec.media_item_images("audio", os.path.join(FIX, "tone.wav")) == [])
check("音频时长 > 0", media_codec.media_duration(os.path.join(FIX, "tone.wav")) > 0)

try:
    media_codec.media_item_images("image", os.path.join(FIX, "not_exist.png"))
    raise AssertionError("应抛异常")
except Exception as e:
    print("PASS: 缺失文件抛异常 (" + type(e).__name__ + ")")

try:
    media_codec.media_item_images("unknown", os.path.join(FIX, "img.png"))
    raise AssertionError("应抛异常")
except ValueError:
    print("PASS: 未知类型抛 ValueError")

print("ALL PASS")
