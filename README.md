# H3PromptOpt —— 工作流式本地提示词优化插件

纯 API 驱动（`/upload` + `/prompt`），本地 GGUF 多模态模型做提示词优化，无第三方在线 API、无人工确认。

## 节点

| 节点 | 作用 |
|---|---|
| `H3_OptMediaInput` | 提示词 + 15 个可选素材文件名控件（图 9 / 视频 3 / 音频 3）→ 请求载荷 |
| `H3_OptLocalRun` | 过滤被引用素材 → 自动任务判定 → 系统提示词 → llama_cpp 生成 → 后处理 → STRING |
| `H3_OptSaveText` | 写 `output/h3promptopt/*.txt` 并原样返回 STRING（备选输出节点） |

> 模板工作流的节点 3 使用**核心 `SaveText`**（`filename_prefix=h3promptopt`）：前端 1.49.6 只为核心
> `SaveText`/`PreviewAny` 节点挂文本预览组件，自定义节点名的 STRING 输出不会在 UI 上渲染。
> `/history` 读取契约不变（`outputs["3"]["text"][0]`），落盘文件在 `output/h3promptopt_*.txt`。

## 使用流程

1. `/upload` 上传图片/视频/音频（每个文件一次），记下返回的文件名；
2. 用 `workflows/prompt_optimize.json` 作模板构造 `/prompt` 请求（只改值、不动结构）：
   - `"1".inputs.prompt`：含 `<Picture n>` / `<Video n>` / `<Audio n>` 占位符的提示词（n 从 1 起，大小写不敏感）；
   - `"1".inputs.picture_1`..`picture_9` / `video_1`..`video_3` / `audio_1`..`audio_3`：`/upload` 返回的文件名（未用留空串）；支持 `subfolder/name` 与绝对路径；
   - `"1".inputs.duration`（秒，默认 5）、`"1".inputs.output_language`（中文/English）；
   - `"2".inputs.model_file` / `mmproj_file` / `device` / `max_tokens` / `keep_loaded`：不填则用默认（默认模型见 `llm_local.py` 顶部 ★ 常量块）；
3. 轮询 `/history/{prompt_id}`，读 `outputs["2"]["text"][0]`（或 `outputs["3"]["text"][0]`）即优化结果；txt 同时落盘。

## 注意事项

- 占位符引用了**未提供**的素材会被静默忽略（与 H3MediaBoard v3 一致）；已提供但未被引用的素材不会送模型；
- 音频只作为文字元数据（文件名+时长）送模型，模型听不到音频内容；
- `keep_loaded=True` 可免去每次 27B 模型加载时间，但显存常驻约 20GB 级，与其他工作流并发有 OOM 风险；
- 本地 27B 生成可达数分钟：`/prompt` 立即返回，客户端轮询 `/history` 需有足够耐心。
- **环境要求**：`llama_cpp` 需 >= 0.3.43 的 CUDA 构建（Qwen3.8-27B 为 3:1 SSM 混合架构，0.3.30 及以下会报 `missing tensor 'blk.*.ssm_conv1d.weight'`；PyPI 新版 wheel 为 CPU-only，本机 CUDA 包在便携版根目录 `whl/` 下，安装：`./python_embeded/python.exe -m pip install --no-deps --force-reinstall --no-index whl/llama_cpp_python-0.3.48+cu130-cp313-cp313-win_amd64.whl`）。

## 换模型

只改 `llm_local.py` 顶部 `MODEL_DEFAULTS` 常量块（相对路径按 `ComfyUI/models/llm` 解析）。支持 qwen3.5/3.6/3.8 与 qwen3-vl 系列。

## 测试

```bash
cd <ComfyUI便携版根目录> && for t in test_core test_codec test_llm test_nodes test_workflow; do
  PYTHONUTF8=1 ./python_embeded/python.exe ComfyUI/custom_nodes/H3PromptOpt/tests/$t.py || exit 1
done
```
