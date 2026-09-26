"""H3PromptOpt 系统提示词核心 —— 硬编码 MinimaxH3 提示词规范。

从 H3MediaBoard prompt_optimizer.py 逐字拷贝（函数体一字不改，保证行为一致）：
_system_prompt / _TAG_RE / _filter_media_by_prompt / _label_num / _auto_task
/ _strip_filenames / _format_prompt_sections / _normalize_subject_shorthand
/ _extract_explicit_dialogues / _ensure_supplied_dialogues / _ensure_fl2va_picture_labels
"""
from __future__ import annotations

import re

def _system_prompt(
    task: str,
    duration: float,
    labels: list[str],
    output_language: str = "English",
    context: dict | None = None,
    user_prompt: str = "",
) -> str:
    task_upper = task.upper()
    fl2va = task_upper == "FL2VA"
    full_reference = task_upper in {"REF2VA", "HYBRID"}
    context = context if isinstance(context, dict) else {}
    label_rule = (
        "FL2VA picture labels must be bare lowercase: picture 1 and picture 2, never angle brackets. "
        "This is a mandatory content requirement, not merely a label-format example: the final prompt must explicitly "
        "contain both exact tokens 'picture 1' and 'picture 2'. The first line must state that picture 1 is the exact "
        f"opening-frame anchor at 0.00s and picture 2 is the exact ending-frame anchor at {duration:.2f}s. Never omit, "
        "rename, translate, capitalize, or replace either token, even when the requested output language is Chinese."
        if fl2va else
        "All media labels must use angle brackets exactly, for example <picture 1>, <video 1>, <audio 1>."
    )
    if full_reference:
        structure = (
            "Return exactly these six sections in this order, with each heading followed by a colon: "
            "subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, "
            "non_diegetic_music. Formatting is strict: every section heading must occupy its own line; put a newline "
            "immediately after the colon, write that section's content on the following line, and put one blank line "
            "between sections. Never place a heading and its content on the same line, and never join multiple sections "
            "into one paragraph. In subject_definitions, define reusable visible content as <Subject N> and cite "
            "its source label using one short sentence and no more than four identifying visual traits; "
            "a media source label may be cited only for content actually visible in that media, and content that "
            "appears only in the user's text must not cite any media label (example: if picture 1 shows only a "
            "red rose while the user prompt also mentions a dancer, the rose subject cites <picture 1> and the "
            "dancer subject carries no source label at all); define "
            "standalone <Picture N> only for a concrete frame anchor. In summary, use one sentence beginning "
            "with a square-bracketed combination of only the relationships that apply: keyframe completion, "
            "reference generation, video editing, video continuation, audio reuse, or audio reference. In "
            "retention_analysis, write one compact line per supplied label and never for any other label, "
            "and use fully_preserved, partially_preserved, "
            "attribute_transfer, or weak_reference for visual content, and fully_copy, partially_copy, reference, "
            "or weak_reference for audio. A relationship marker alone is never a complete entry. Every line must use "
            "the exact form '<label>: marker - explanation' and add a concise, concrete explanation after the hyphen. "
            "For fully_preserved, state which identity, appearance, scene, composition, or structural properties remain "
            "unchanged in the target. For attribute_transfer, state exactly which appearance, motion, scene, camera, "
            "timing, or structural attributes transfer from the reference and which target receives them. For audio, "
            "state whether the complete signal is copied unchanged, only selected portions or layers are copied, or "
            "only timbre, rhythm, delivery, or other specified attributes are referenced. Use enough information to "
            "make each reference relationship operational, normally one full sentence per label, but do not repeat an "
            "exhaustive appearance inventory already stated in subject_definitions. Never output bare entries such as "
            "'<picture 1>: fully_preserved', '<video 1>: attribute_transfer', or '<audio 1>: fully_copy'. "
            "Make detailed_description action-first, chronological, and production-ready. Give the scene a clear "
            "dramatic progression—establishment, development, a meaningful visual or emotional beat, and a resolved "
            "ending—without changing the user's intended event. Describe subject blocking, observable state changes, "
            "composition, camera movement, and the final visual landing. Use [Shot 1] and timed later shots only for "
            "motivated real cuts that reveal new information; otherwise design a fluid continuous shot. Normally allow "
            "about 350-700 Chinese characters or 220-450 English words for the complete six-section result, and use "
            "more when duration, dialogue, multiple subjects, or genuine shot complexity requires it."
        )
    else:
        structure = (
            "Return exactly these three sections in order: integrated_multimodal_description, overall_soundscape, "
            "non_diegetic_music. Formatting is strict: every section heading must occupy its own line; put a newline "
            "immediately after the colon, write that section's content on the following line, and put one blank line "
            "between sections. Never place a heading and its content on the same line. Build a complete audiovisual "
            "progression rather than merely paraphrasing the user's sentence. Use [Shot 1] for the opening and "
            "timestamps only for motivated real later cuts. For I2VA, "
            "anchor the supplied picture at 0.00s and develop forward. For L2VA, infer a plausible opening and "
            "converge exactly to the supplied picture at the end. For FL2VA, describe one continuous, observable "
            "motion path from picture 1 at 0.00s to picture 2 at the target end time. Give the main description enough "
            "detail to establish the opening composition, action onset, intermediate development, camera evolution, "
            "and a visually resolved ending within the supplied duration. In FL2VA, mention picture 1 again when "
            "establishing the opening state and picture 2 again when describing the final visual landing; both labels "
            "are compulsory and must not be replaced by generic phrases such as 'the first image' or 'the last image'."
        )

    keyframe_lines = []
    for item in context.get("keyframes") or []:
        if not isinstance(item, dict) or not item.get("label") or not item.get("role"):
            continue
        role_text = {
            "exact_first_frame": "is the exact first-frame anchor at 0.00s",
            "exact_last_frame": f"is the exact last-frame anchor at {duration:.2f}s",
            "keyframe": "is a concrete visual keyframe anchor",
        }.get(str(item["role"]), "is a visual reference")
        keyframe_lines.append(f"{item['label']} {role_text}")
    keyframe_rule = (
        "Keyframe roles supplied by the node: " + "; ".join(keyframe_lines) + ". Preserve these roles exactly."
        if keyframe_lines else
        "No concrete keyframe role was supplied by the node; do not invent one."
    )

    audio_mode = str(context.get("audio_mode") or "native")
    # 开头明确带出模式 token（如 reference_only），便于模型精确遵循所选音频模式
    audio_rule = f"Audio mode={audio_mode}. " + {
        "lock_source": (
            "Audio mode is original-audio output. State only that the source video's original audio is preserved "
            "unchanged as the complete target soundtrack and mark it fully_copy. Because the optimizer does not "
            "receive or hear the audio binary, do not identify, describe, classify, or invent any dialogue, music, "
            "ambience, sound effect, instrument, language, rhythm, or other audible content. In overall_soundscape, "
            "write only the selected-language equivalent of 'Preserve the source video's original audio unchanged.' "
            "Do not add any new sound, voice, animal vocalization, ambience, or music."
        ),
        "reference_only": (
            "Audio mode is reference-only: treat supplied audio as audio reference; do not claim its waveform "
            "is copied into the final soundtrack."
        ),
        "remix_source": (
            "Audio mode is source-audio remix: describe only partial copying or remixing of the supplied audio, "
            "never a 1:1 complete copy."
        ),
        "native": "Audio mode is automatic generation: do not claim any uploaded audio is copied into the final output.",
    }.get(audio_mode, "Describe the audio relationship conservatively and do not invent how it is used.")

    singing_rule = ""
    if re.search(
        r"唱|歌曲|歌词|歌唱|演唱|口型|嘴型|对口型|lip\s*sync|sing(?:ing|s)?|lyrics?|vocal",
        user_prompt,
        flags=re.IGNORECASE,
    ):
        singing_rule = (
            " The request involves singing or lip synchronization. Require accurate phoneme-level lip and jaw "
            "synchronization throughout the vocals: mouth movement should follow syllable onset and release, clear "
            "consonant closures, natural vowel shapes, pauses and breaths, and sustained notes; the mouth rests "
            "naturally during non-vocal spans. Also require stable facial identity and continuous mouth motion without "
            "generic loops, visible delay, frozen lips, or facial deformation. Do not reduce a singing performance "
            "to lip movement while the rest of the body remains still. Unless the user explicitly requests restrained "
            "or static performance, design evolving full-body or upper-body performance across the timeline: shifting "
            "weight, torso and shoulder rhythm, head movement, gaze changes, expressive but natural hand and arm "
            "phrasing, and transitions between distinct gestures. Make these movements non-repetitive and responsive "
            "to musical phrases, accents, pauses, intensity changes, and vocal emotion without inventing exact unheard "
            "beats. Coordinate purposeful camera motion and changing shot emphasis with the performance so the image "
            "does not feel like a static portrait with moving lips. Keep lip-sync requirements compact, but give the "
            "visible performance and camera progression enough concrete detail to guide a lively result. Because the audio binary is not "
            "transmitted to the optimizer, never invent exact lyrics, notes, beats, or timestamps and never claim "
            "to have analyzed or heard the file."
        )
    transfer_rule = ""
    if re.search(
        r"替换|置换|换成|变成|动作迁移|模仿.*动作|复刻.*动作|跟随.*动作|"
        r"replace|replacement|swap|motion\s*(?:transfer|reference)|imitate.*motion|copy.*motion",
        user_prompt,
        flags=re.IGNORECASE,
    ):
        transfer_rule = (
            " This is a subject-replacement or motion-transfer request. The following is a strict task-specific "
            "exception that overrides the general instructions to describe actions, timing, performance, camera, "
            "sound, and shot-by-shot events. Describe only the source-to-target relationship: which subject from "
            "<picture N> replaces which subject in <video N>, while the source video's complete motion, timing, "
            "scene, props, composition, camera movement, cuts, and temporal structure remain unchanged. Never name, "
            "infer, enumerate, paraphrase, or timestamp any concrete source-video action, gesture, pose change, facial "
            "expression, dialogue, object interaction, prop function, story event, sound, ambience, or music, even if "
            "it appears visible or inferable in the sampled frames. Use the sampled frames only to distinguish the "
            "subjects that must be replaced. Define replacement subjects only with official numbered labels such as "
            "<Subject 1> and <Subject 2>, never semantic labels such as <cat> or a Chinese name. Give each subject only "
            "the minimum clearly visible identity cues needed to avoid mismatch: subject type or gender, approximate "
            "age when reliable, main clothing and color, hairstyle or fur pattern, hat, and glasses. Do not infer an "
            "occupation or describe pose, expression, action, scene details, lighting, or unrelated appearance. Do not "
            "define the original subjects being removed as target <Subject N> entries unless needed only to make an "
            "unambiguous one-to-one replacement mapping. In detailed_description, keep a single [Shot 1] for the source "
            "timeline, but write a complete operational paragraph rather than a bare one-sentence relationship. State "
            "which source subject or scene is replaced by which referenced target, which target identity and major "
            "appearance or environment attributes remain stable, and which source-video properties remain unchanged: "
            "motion performance, facial-expression timing, blocking, screen direction, camera trajectory, framing, "
            "cuts, pacing, and duration when applicable. Explain how replaced subjects remain spatially integrated with "
            "preserved props and scene elements, but never invent or enumerate the source video's concrete actions, "
            "dialogue, props, events, or sounds; do not expand the source timeline or reconstruct it from sampled frames. Treat the three sampled images from one <video N> as observation "
            "frames of that single reference video, never as separate target shots or evidence of cuts."
        )
    # 任务感知的官方段名清单：T2VA 三段式不得提及六段式专属字段名
    section_names_cn = (
        "subject_definitions、summary、retention_analysis、detailed_description、"
        "integrated_multimodal_description、overall_soundscape、non_diegetic_music"
        if full_reference else
        "integrated_multimodal_description、overall_soundscape、non_diegetic_music"
    )
    language_rule = (
        "输出语言强制为中文：所有自然语言说明、主体描述、摘要、镜头描述、声音说明和音乐说明必须使用简体中文，"
        f"不得输出英文句子。只有官方固定字段名 {section_names_cn}，"
        "媒体与主体标签、[Shot N]，以及 fully_preserved、partially_preserved、attribute_transfer、"
        "weak_reference、fully_copy、partially_copy、reference 等固定关系标记保留英文。返回前必须检查并将"
        "其他英文自然语言全部改写成中文。"
        if output_language == "中文" else
        "Output all natural-language content in English. Keep supplied dialogue, lyrics, and visible text in their "
        "original language."
    )
    # 任务感知的 <d> 禁止段：只禁止当前任务实际存在的非主体描述段
    forbidden_sections = (
        "subject_definitions, summary, retention_analysis, overall_soundscape, or non_diegetic_music"
        if full_reference else
        "overall_soundscape or non_diegetic_music"
    )
    dialogue_rule = (
        "Official dialogue formatting is mandatory in every task and is independent of the selected narration "
        "language. Put every user-supplied spoken line or lyric inside exactly one <d>[Language]...</d> block, using "
        "the dialogue's actual language name in English. In particular, Chinese speech must use the exact form "
        "<d>[Chinese]中文原文</d>; never write Chinese dialogue only inside quotation marks, and never omit [Chinese]. "
        "Keep the speaker description, stable speaker ID such as (S1), speaking action, and delivery outside the <d> "
        "block; inside it preserve only the user's exact words and punctuation without translation or rewriting. "
        "The output-language setting controls narration only: an English prompt must still retain supplied Chinese "
        "dialogue as <d>[Chinese]...</d>. Placement is as mandatory as formatting: every <d> block must be embedded "
        "inside integrated_multimodal_description or detailed_description at the exact chronological moment when the "
        "speaker speaks or sings, immediately after that speaker's speaking/singing action and delivery description. "
        "Never collect dialogue into a list, appendix, footer, or separate paragraph after the official sections. "
        f"Never place a <d> block in {forbidden_sections}. The final section is always non_diegetic_music, "
        "and absolutely no text or <d> block may "
        "follow that section's content. For multiple supplied lines, preserve their original order and place each line "
        "at its corresponding point in the shot timeline, with intervening actions and pauses described between lines "
        "when the user's request implies them. Before returning, verify that every supplied line appears exactly once "
        "inside the main shot timeline and nowhere else. Do not wrap paraphrased soundscape descriptions, ambient "
        "sounds, or dialogue that the user did not explicitly supply in <d> tags."
    )
    subject_shorthand_rule = (
        "Subject shorthand formatting is mandatory and globally consistent. If a stable short subject ID is used, "
        "it must always be written with parentheses as (S1), (S2), ... through (S20). Never output a bare S1, S2, "
        "or any other unparenthesized S-number in narration, shot descriptions, actions, dialogue attribution, or "
        "sound descriptions. <Subject N> definitions remain unchanged; (S1) is only the shorthand used to refer back "
        "to a defined subject. Before returning, scan the complete result and replace every standalone bare S1-S20 "
        "with its exact parenthesized form."
    )
    return (
        "You are a professional prompt writer for the open-source MiniMax H3 audiovisual model. Return only the "
        "final production-ready prompt without explanations, markdown fences, filenames, or invented media. "
        "Preserve the user's intent and every supplied dialogue, lyric, and visible-text word verbatim; do not "
        "invent dialogue or lyrics that were not provided. This is a video-generation prompt, not image captioning "
        "or visual reverse-prompting. Reference images are already passed into model conditioning, so identify each "
        "visual subject only with the minimum distinctive cues needed to disambiguate it: usually gender or subject "
        "type, main clothing, main hairstyle, and scene. Do not describe facial features, lighting, pose, background "
        "objects, or composition exhaustively unless they are essential to the requested motion or must change over "
        "time. Spend most words on what happens after the reference frame: causally connected actions, reactions, "
        "story progression, timing, performance, camera behavior, and audio-visual synchronization. Develop the "
        "user's idea into concrete visible beats that fit the duration: establish the situation, let the central "
        "action evolve through meaningful intermediate changes, add a restrained climax or reveal when appropriate, "
        "and finish on a clear visual result rather than an abrupt stop. Do not introduce unrelated characters, props, "
        "locations, conflicts, or plot twists. Avoid repeating the same static trait across sections, avoid decorative "
        "adjectives, and never turn quality requirements into a long negative-prompt checklist. Use professional but "
        "executable cinematography with dynamic video direction by default. Unless the user explicitly requests a static pose, locked-off camera, "
        "minimal movement, or a specific fixed composition, do not let the subject remain nearly motionless after the "
        "opening frame. Build visible motion in successive phases with clear variation and continuity: changes in body "
        "weight, posture, orientation, gesture, interaction, expression, spatial position, or object state. Avoid one "
        "gesture repeated mechanically throughout the clip. For music, singing, dance, performance, fashion, or other "
        "rhythmic scenes, let the body express phrase changes through varied natural gestures and coordinated torso, "
        "shoulder, head, hand, and footwork where framing permits; align movement energy and camera emphasis with broad "
        "musical progression without fabricating unheard exact beats. Treat the reference image as a starting anchor, "
        "not the main subject of the description: spend only enough text to identify it, then prioritize motion design, "
        "performance evolution, kinetic atmosphere, and a visually active ending. For each shot, integrate the useful framing or shot scale, subject blocking, and "
        "camera path into the action. When motion benefits the scene, choose a motivated Push In, Pull Out, Pan, Tilt, "
        "Truck, Pedestal, Arc Shot, Tracking Shot, or controlled Zoom, and specify subtle/large amplitude and slow/fast "
        "speed only when meaningful. Let camera movement reveal information, follow motion, emphasize a transformation, "
        "or land on the final state; do not stack random camera terms or use constant movement without purpose. Vary "
        "composition and visual emphasis over time so the video feels directed rather than static, while preserving "
        "spatial continuity, screen direction, subject identity, and reference constraints. Expand into continuous "
        "actions, camera choreography, synchronized audible events, ambience, and music with enough detail to guide "
        "generation effectively. Prefer a well-designed continuous shot when it can express the event clearly, but "
        "allow a small number of motivated cuts when they materially improve narrative clarity or reveal new visual "
        "information. Do not create labels outside the supplied label list, renumber labels, "
        "or replace labels with filenames. "
        f"Task={task}; duration={duration:.2f}s; available labels={', '.join(labels) or 'none'}. "
        f"{label_rule} {keyframe_rule} {audio_rule} {structure}{singing_rule}{transfer_rule} {dialogue_rule} "
        f"{subject_shorthand_rule} "
        "Audio items are label-only references; never claim you listened to them. Before returning, silently check "
        "that every required section is present in the correct order, every used label exists, no filename appears, "
        "no audio content was fabricated, all timing fits the target duration, and the audio relationship matches "
        f"the supplied mode. {language_rule}"
    )

_TAG_RE = re.compile(r"<(picture|video|audio)\s*(\d{1,2})\s*>", re.IGNORECASE)


def _filter_media_by_prompt(prompt: str, media: list) -> list:
    """按提示词中的 <Picture n>/<Video n>/<Audio n> 标签收集被引用的素材。
    大小写不敏感；重复标签去重；标签引用不存在的 label 忽略；保持 media 原顺序。
    标签类别 picture 与素材 kind="image" 等价。"""
    wanted = {(m.group(1).lower(), int(m.group(2))) for m in _TAG_RE.finditer(prompt or "")}
    if not wanted:
        return []
    result = []
    for item in media or []:
        kind = str(item.get("kind") or "").lower()
        num = _label_num(item)
        if (kind, num) in wanted or (kind == "image" and ("picture", num) in wanted):
            result.append(item)
    return result


def _label_num(item: dict) -> int:
    m = re.search(r"(\d+)\s*$", str(item.get("label") or ""))
    return int(m.group(1)) if m else -1


def _auto_task(prompt: str, media: list) -> str:
    """自动任务判定：提示词引用任何素材 -> HYBRID（六段式）；否则 T2VA（三段式）。"""
    return "HYBRID" if _filter_media_by_prompt(prompt, media) else "T2VA"

def _strip_filenames(text: str) -> str:
    return re.sub(
        r"(?<![\w/\\])[\w .()\-\u4e00-\u9fff]+\.(?:png|jpe?g|webp|bmp|gif|mp4|mov|webm|mkv|avi|mp3|wav|flac|m4a|ogg|aac)(?!\w)",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()


def _format_prompt_sections(text: str) -> str:
    """Force official section headings onto separate lines without rewriting content."""
    headings = (
        "subject_definitions",
        "summary",
        "retention_analysis",
        "detailed_description",
        "integrated_multimodal_description",
        "overall_soundscape",
        "non_diegetic_music",
    )
    pattern = r"\s*(" + "|".join(map(re.escape, headings)) + r")\s*:\s*"
    formatted = re.sub(pattern, lambda match: f"\n\n{match.group(1).lower()}:\n", str(text), flags=re.IGNORECASE)
    return formatted.strip()


def _normalize_subject_shorthand(text: str) -> str:
    """Guarantee that standalone S1-S20 references use official parentheses."""
    return re.sub(
        r"(?<![A-Za-z0-9_(<（])S([1-9]|1\d|20)(?![A-Za-z0-9_)>）])",
        lambda match: f"(S{match.group(1)})",
        str(text),
        flags=re.IGNORECASE,
    )


def _extract_explicit_dialogues(prompt: str) -> list[tuple[str, str]]:
    """Extract only dialogue that the user explicitly supplied, preserving its exact text."""
    source = str(prompt or "")
    candidates = []
    speech_marker = r"(?:说|说道|说着|喊|喊道|问|问道|回答|答道|台词|对白|says?|speaks?|shouts?|asks?|replies?)"
    patterns = (
        rf"{speech_marker}[^\n“”‘’\"']{{0,20}}[：:]?\s*[“\"]([^”\"\n]+)[”\"]",
        rf"{speech_marker}[^\n“”‘’\"']{{0,20}}[：:]?\s*[‘']([^’'\n]+)[’']",
        rf"{speech_marker}\s*[：:]\s*([^\n；;]+)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, source, flags=re.IGNORECASE):
            value = match.group(1).strip().strip("“”‘’\"'").strip()
            existing = {item[0] for item in candidates}
            if not value or value in existing or any(value in item or item in value for item in existing):
                continue
            language = "Chinese" if re.search(r"[\u3400-\u9fff]", value) else "English"
            candidates.append((value, language))
    return candidates


def _ensure_supplied_dialogues(text: str, user_prompt: str, output_language: str) -> str:
    """Keep supplied dialogue exactly once and inside the main shot timeline."""
    result = str(text)
    dialogues_to_insert = []
    for index, (dialogue, language) in enumerate(_extract_explicit_dialogues(user_prompt), start=1):
        tag = f"<d>[{language}]{dialogue}</d>"
        main_heading = re.search(
            r"(?:integrated_multimodal_description|detailed_description)\s*:\s*",
            result,
            flags=re.IGNORECASE,
        )
        main_start = main_heading.end() if main_heading else 0
        main_end_match = re.search(
            r"(?:overall_soundscape|non_diegetic_music)\s*:\s*",
            result[main_start:],
            flags=re.IGNORECASE,
        )
        main_end = main_start + main_end_match.start() if main_end_match else len(result)
        timeline = result[main_start:main_end]
        if tag in timeline:
            # Remove accidental duplicates outside the timeline while preserving the valid occurrence.
            prefix = result[:main_start].replace(tag, "")
            suffix = result[main_end:].replace(tag, "")
            result = prefix + timeline + suffix
            continue
        # A correctly tagged line outside the shot timeline is structurally invalid; move rather than duplicate it.
        result = result.replace(tag, "")
        main_heading = re.search(
            r"(?:integrated_multimodal_description|detailed_description)\s*:\s*",
            result,
            flags=re.IGNORECASE,
        )
        main_start = main_heading.end() if main_heading else 0
        main_end_match = re.search(
            r"(?:overall_soundscape|non_diegetic_music)\s*:\s*",
            result[main_start:],
            flags=re.IGNORECASE,
        )
        main_end = main_start + main_end_match.start() if main_end_match else len(result)
        timeline = result[main_start:main_end]
        if dialogue in timeline:
            replaced = False
            for quoted in (f"“{dialogue}”", f"‘{dialogue}’", f'"{dialogue}"', f"'{dialogue}'"):
                if quoted in timeline:
                    timeline = timeline.replace(quoted, tag, 1)
                    replaced = True
                    break
            if not replaced:
                timeline = timeline.replace(dialogue, tag, 1)
            result = result[:main_start] + timeline + result[main_end:]
            continue
        dialogues_to_insert.append((index, tag))

    if dialogues_to_insert:
        main_heading = re.search(
            r"(?:integrated_multimodal_description|detailed_description)\s*:\s*",
            result,
            flags=re.IGNORECASE,
        )
        main_start = main_heading.end() if main_heading else 0
        main_end_match = re.search(
            r"(?:overall_soundscape|non_diegetic_music)\s*:\s*",
            result[main_start:],
            flags=re.IGNORECASE,
        )
        main_end = main_start + main_end_match.start() if main_end_match else len(result)
        timeline = result[main_start:main_end]
        cue = re.search(
            r"(?:开口说话|开始说话|随后说|继续说|说话|演唱|唱着|begins? speaking|starts? speaking|speaks?|says?|sings?)",
            timeline,
            flags=re.IGNORECASE,
        )
        if cue:
            boundary = re.search(r"[。！？.!?](?:\s|$)", timeline[cue.end():])
            insert_at = cue.end() + boundary.end() if boundary else cue.end()
        else:
            shot = re.search(r"\[Shot 1\]\s*", timeline, flags=re.IGNORECASE)
            insert_at = shot.end() if shot else 0
        sentences = []
        for order, (index, tag) in enumerate(dialogues_to_insert):
            if str(output_language).lower() in {"中文", "chinese", "zh", "zh-cn"}:
                action = "说" if order == 0 else "随后继续说"
                sentences.append(f"画面中的说话者 (S{index}) {action}：{tag}")
            else:
                action = "says" if order == 0 else "then continues"
                sentences.append(f"The on-screen speaker (S{index}) {action}: {tag}")
        insertion = " " + " ".join(sentences) + " "
        timeline = timeline[:insert_at] + insertion + timeline[insert_at:]
        result = result[:main_start] + timeline + result[main_end:]
    return re.sub(r"[ \t]+\n", "\n", result).strip()


def _ensure_fl2va_picture_labels(text: str, task: str, duration: float, output_language: str) -> str:
    """Guarantee that an FL2VA result retains both official bare picture labels."""
    if str(task).upper() != "FL2VA":
        return text
    lowered = str(text).lower()
    if "picture 1" in lowered and "picture 2" in lowered:
        return text
    if str(output_language).lower() in {"中文", "chinese", "zh", "zh-cn"}:
        alignment = (
            f"参考图像与目标视频对齐关系：picture 1 对齐目标视频的 0.00 秒首帧；"
            f"picture 2 对齐目标视频的 {duration:.2f} 秒尾帧。"
        )
    else:
        alignment = (
            "Reference-picture alignment: picture 1 is the target video's exact opening frame at 0.00s; "
            f"picture 2 is its exact ending frame at {duration:.2f}s."
        )
    return f"{alignment}\n\n{text}".strip()




# ===================== H3PromptOpt 自增后处理（非 v3 拷贝） =====================
def clean_retention_analysis(text: str, labels: list[str]) -> str:
    """清洗 retention_analysis 小节：只保留「已提供标签」的行，同一标签仅保留第一行。

    小模型（9B）会捏造未提供的标签行（如 <audio 1>）或对同一标签重复多行；
    该清洗是确定性兜底，不受模型执行质量影响。非标签行（解释文字/其他小节标题）原样保留。
    """
    allowed = {str(label).strip().lower() for label in (labels or []) if label}
    if not allowed:
        return text
    lines = str(text).splitlines()
    out: list[str] = []
    seen: set[str] = set()
    in_retention = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"^retention_analysis\s*:\s*$", stripped, flags=re.IGNORECASE):
            in_retention = True
            out.append(line)
            continue
        if not in_retention:
            out.append(line)
            continue
        if not stripped:  # 空行 = 小节结束
            in_retention = False
            out.append(line)
            continue
        match = re.match(r"^<([a-z]+)\s*\d+>\s*:", stripped, flags=re.IGNORECASE)
        if not match:  # 非标签行保留
            out.append(line)
            continue
        num_match = re.search(r"(\d+)\s*>", stripped)
        key = f"{match.group(1).lower()} {int(num_match.group(1))}" if num_match else None
        if key is not None and key not in allowed:
            continue  # 未提供的标签行：丢弃
        if key is not None and key in seen:
            continue  # 同一标签重复：丢弃
        if key is not None:
            seen.add(key)
        out.append(line)
    return "\n".join(out)
