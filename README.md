# H3PromptOpt
纯 API 驱动的 ComfyUI 本地提示词优化工作流：/upload 上传图片/视频/音频 → /prompt 提交模板工作流（提示词含 &lt;Picture n> / &lt;Video n> / &lt;Audio n> 占位符，最多 9 图 / 3 视频 / 3 音频）→ 本地 GGUF 多模态模型优化提示词 → 六段式 MinimaxH3 规范输出，UI 可见 + 落盘 txt + /history 可读。
