import re
import requests
import json
import os
import time
import base64
from typing import Dict, Tuple, List
from PIL import Image
import io

# 缓存：key 为元组，值为字符串
_CACHE: Dict[tuple, str] = {}
_CONFIG = None
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "ai_config.json")

def _load_config():
    default = {
        "base_url": "http://127.0.0.1:1234",
        "token": "",
        "timeout": 60,
        "last_model": ""
    }
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                user = json.load(f)
                default.update(user)
    except:
        pass
    return default

def _save_config(base_url=None, token=None, model=None):
    global _CONFIG
    try:
        cfg = {}
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        if base_url:
            cfg["base_url"] = base_url
        if token:
            cfg["token"] = token
        if model:
            cfg["last_model"] = model
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        _CONFIG = _load_config()
    except:
        pass

_CONFIG = _load_config()

def _clean_thinking_response(text: str) -> str:
    """清理响应中的思考过程，只保留最终结果"""
    if not text:
        return text
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^(Thinking Process:|思考过程：|【思考】|\[思考\]|让我想想|我先|我需要|好的，|好，|明白了|理解了|收到|行，|嗯，|综上所述|总结|我认为|我觉得|分析：|思考：|推理：|解读：|Let me think|First|Then|Next|Finally|I think|Analysis:|Reasoning:)', stripped, re.IGNORECASE):
            continue
        cleaned.append(line)
    result = '\n'.join(cleaned).strip()
    return result if result else text.strip()

def _parse_marked_output(text: str) -> dict:
    """解析带标记的输出格式，返回 {positive, negative, description}"""
    result = {"positive": "", "negative": "", "description": ""}
    if not text:
        return result
    positive_match = re.search(r'\[POSITIVE\](.*?)(?=\[NEGATIVE\]|$)', text, re.DOTALL)
    negative_match = re.search(r'\[NEGATIVE\](.*?)(?=\[DESCRIPTION\]|$)', text, re.DOTALL)
    description_match = re.search(r'\[DESCRIPTION\](.*?)$', text, re.DOTALL)
    if positive_match:
        result["positive"] = positive_match.group(1).strip()
    if negative_match:
        result["negative"] = negative_match.group(1).strip()
    if description_match:
        result["description"] = description_match.group(1).strip()
    return result

def _ai_chat(url: str, token: str, model: str, msg: str, timeout: int = 60, image_base64: str = None) -> str:
    """通用的AI聊天接口，支持OpenAI兼容的API，支持多模态"""
    base = url.rstrip('/')
    final_msg = msg.strip() + "\n\n/no_think"
    if image_base64:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": final_msg},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        }]
        payload = {"model": model, "messages": messages, "temperature": 0.1}
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            r = requests.post(f"{base}/v1/chat/completions", headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    res = choice["message"]["content"].strip()
                    return _clean_thinking_response(res)
                elif "text" in choice:
                    return _clean_thinking_response(choice["text"].strip())
                elif "content" in choice:
                    return _clean_thinking_response(choice["content"].strip())
            if "output" in data and isinstance(data["output"], list):
                for item in data["output"]:
                    if item.get("type") == "message" and "content" in item:
                        return _clean_thinking_response(item["content"].strip())
                for item in data["output"]:
                    if "content" in item:
                        return _clean_thinking_response(item["content"].strip())
            if "response" in data:
                return _clean_thinking_response(data["response"].strip())
            return str(data)
        except Exception as e:
            pass
    endpoints = [
        {"url": f"{base}/api/v1/chat", "payload": {"model": model, "input": final_msg, "temperature": 0.1}},
        {"url": f"{base}/v1/chat/completions", "payload": {"model": model, "messages": [{"role": "user", "content": final_msg}], "temperature": 0.1}}
    ]
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for ep in endpoints:
        try:
            r = requests.post(ep["url"], headers=headers, json=ep["payload"], timeout=timeout)
            r.raise_for_status()
            data = r.json()
            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    res = choice["message"]["content"].strip()
                elif "text" in choice:
                    res = choice["text"].strip()
                elif "content" in choice:
                    res = choice["content"].strip()
                else:
                    res = str(choice).strip()
            elif "response" in data:
                res = data["response"].strip()
            elif "output" in data and isinstance(data["output"], list):
                for item in data["output"]:
                    if item.get("type") == "message" and "content" in item:
                        res = item["content"].strip()
                        break
                else:
                    if "content" in data["output"][0]:
                        res = data["output"][0]["content"].strip()
                    else:
                        res = str(data["output"][0]).strip()
            else:
                res = str(data).strip()
            res = _clean_thinking_response(res)
            _save_config(model=model)
            return res
        except:
            continue
    raise Exception("连接失败")

# ---------- 图片提示词 ----------
def _format_image_prompt(manual_text: str, optional_text: str, mode: str,
                         output_type: str, detail_level: str, output_lang: str) -> str:
    has_manual = bool(manual_text and manual_text.strip())
    has_optional = bool(optional_text and optional_text.strip())

    # 任务类型
    if has_optional and has_manual:
        task_type = "both"
    elif has_optional:
        task_type = "only_optional"
    else:
        task_type = "only_manual"

    # 数量建议
    if detail_level == "标准":
        if output_type == "词汇":
            min_pos, min_neg = 60, 20
        else:
            min_pos_chars, min_neg_chars = 200, 80
    elif detail_level == "详细":
        if output_type == "词汇":
            min_pos, min_neg = 120, 30
        else:
            min_pos_chars, min_neg_chars = 400, 150
    else:  # 极详细
        if output_type == "词汇":
            min_pos, min_neg = 200, 40
        else:
            min_pos_chars, min_neg_chars = 600, 200

    lines = []
    lines.append("【指令】直接输出正向和负向提示词。严禁思考、分析、解释。")
    lines.append("严禁使用Markdown格式。")
    lines.append("")

    # 角色描述
    if output_lang == "中文":
        lines.append(f"你是一个专业的AI绘画提示词专家，使用中文输出，{'文生图' if mode=='文生图' else '图生图'}模式。")
    else:
        lines.append(f"You are a professional AI image prompt expert. Use English output, {'text-to-image' if mode=='文生图' else 'image-to-image'} mode.")
    lines.append("")

    # ==================== 词汇模式 ====================
    if output_type == "词汇":
        if output_lang == "中文":
            lines.append("【词汇模式规则】")
            lines.append("- 输出纯中文关键词，仅允许 8K/4K/UHD 等通用简短的缩略词。")
            lines.append("- 使用英文逗号+空格分隔。")
            lines.append("- 权重语法：英文半角括号 ( ) 提升，[ ] 降低，:数值精确控制。")
            lines.append("- 语序：主体 → 动作 → 场景 → 构图 → 风格 → 画质。")
        else:
            lines.append("【Vocabulary Mode Rules】")
            lines.append("- Output English keywords, comma+space separated.")
            lines.append("- Weight syntax: ( ) up, [ ] down, :value exact.")
            lines.append("- Word order: subject → action → scene → composition → style → quality.")
        lines.append("")

        # 强制格式要求
        lines.append("【必须遵守的输出格式】")
        lines.append("你必须输出两行，每行以方括号标签开头，后面紧跟内容（不要换行，不要加额外空格）：")
        lines.append("第一行：[POSITIVE] 你的正向提示词")
        lines.append("第二行：[NEGATIVE] 你的负向提示词")
        lines.append("")
        lines.append("示例（中文）：")
        lines.append("[POSITIVE] 红裙女孩, 花园浇花, 手持水壶, 阳光, 8K")
        lines.append("[NEGATIVE] 模糊, 低质量, 噪点, 畸形手, 多余肢体")
        lines.append("")
        if output_lang == "中文":
            lines.append("**绝对禁止输出不带标签的内容、JSON、Markdown、解释文字。**")
            lines.append(f"**负向提示词不能为空，至少包含 {min_neg} 个以上常见瑕疵词。**")
        else:
            lines.append("**DO NOT output anything without labels, JSON, Markdown, or explanations.**")
            lines.append(f"**Negative prompt MUST NOT be empty; include at least {min_neg} common defect keywords.**")
        lines.append("")

        # 任务输入（词汇模式）
        if task_type == "both":
            if output_lang == "中文":
                lines.append("【原图描述】" + optional_text.strip())
                lines.append("【手工提示词】" + manual_text.strip())
                lines.append("规则：手工提示词优先替换冲突部分，不冲突元素保留融合。")
            else:
                lines.append("Original: " + optional_text.strip())
                lines.append("Manual: " + manual_text.strip())
                lines.append("Rule: Manual overrides conflicts; keep non-conflicting elements.")
        elif task_type == "only_optional":
            if output_lang == "中文":
                lines.append("【原图描述】" + optional_text.strip())
                lines.append("规则：提取所有视觉元素，添加细节，补充场景。")
            else:
                lines.append("Original: " + optional_text.strip())
                lines.append("Rule: Extract all elements, add details, expand scene.")
        else:  # only_manual
            if output_lang == "中文":
                lines.append("【手工提示词】" + manual_text.strip())
                lines.append("规则：提取核心元素，添加细节，构建合理场景。")
            else:
                lines.append("Manual: " + manual_text.strip())
                lines.append("Rule: Extract core elements, add details, build scene.")

        lines.append("")
        if output_lang == "中文":
            lines.append(f"建议关键词数量：正向至少 {min_pos} 个词，负向至少 {min_neg} 个词。")
        else:
            lines.append(f"Suggested word count: positive at least {min_pos}, negative at least {min_neg}.")
        lines.append("")
        lines.append("现在严格按照格式输出（两行，每行以标签开头）：")
        lines.append("[POSITIVE] ")
        lines.append("[NEGATIVE] ")
        return "\n".join(lines)

    # ==================== 整句模式 ====================
    else:
        if output_lang == "中文":
            lines.append("【整句模式规则】输出自然语言段落。")
            lines.append("")
            lines.append("【语言要求】必须输出纯中文。仅允许以下英文缩略词：8K, 4K, UHD, HDR, AI, PBR。")
            lines.append("除此之外，禁止出现任何英文单词（包括 cinematic, photorealistic, Unreal Engine 等）。")
            lines.append("")
            lines.append("格式：")
            lines.append(f"[POSITIVE] 你的自然语言描述（不少于{min_pos_chars}字符）")
            lines.append(f"[NEGATIVE] 你的自然语言负向描述（不少于{min_neg_chars}字符）")
            lines.append("")
            lines.append("示例：")
            lines.append("[POSITIVE] 一个穿红裙的女孩在花园里浇花，手持绿色水壶，水珠洒在玫瑰上。阳光明媚，蝴蝶飞舞。中景，写实风格，8K超高清。")
            # 修改为完整句子
            lines.append("[NEGATIVE] 负向提示词应避免出现模糊、低画质、畸形手指、多余肢体、噪点等常见画面瑕疵。")
            lines.append("")
        else:
            lines.append("【Sentence Mode Rules】Output natural language paragraphs.")
            lines.append("Format:")
            lines.append(f"[POSITIVE] your description (at least {min_pos_chars} words)")
            lines.append(f"[NEGATIVE] your negative description (at least {min_neg_chars} words)")
            lines.append("Example:")
            lines.append("[POSITIVE] A girl in a red dress waters flowers in a garden, holding a green watering can. Water droplets fall on roses. Sunlight, butterflies. Medium shot, realistic, 8K.")
            lines.append("[NEGATIVE] Avoid blurry, low quality, deformed fingers, extra limbs, noise, and other common defects.")
            lines.append("")

        if task_type == "both":
            if output_lang == "中文":
                lines.append("【任务】综合以下内容，创作高质量的描述。")
                lines.append("【原图描述】" + optional_text.strip())
                lines.append("【手工提示词】" + manual_text.strip())
                lines.append("规则：手工提示词优先替换冲突部分，不冲突元素保留融合。输出自然语言段落。")
            else:
                lines.append("Task: Combine the following. Manual overrides conflicts.")
                lines.append("Original: " + optional_text.strip())
                lines.append("Manual: " + manual_text.strip())
        elif task_type == "only_optional":
            if output_lang == "中文":
                lines.append("【任务】基于原图描述生成描述。")
                lines.append("【原图描述】" + optional_text.strip())
                lines.append("规则：提取所有视觉元素，添加细节，补充场景。输出自然语言段落。")
            else:
                lines.append("Task: Generate from original description only.")
                lines.append("Original: " + optional_text.strip())
        else:  # only_manual
            if output_lang == "中文":
                lines.append("【任务】根据手工提示词生成描述（无原图参考）。")
                lines.append("【手工提示词】" + manual_text.strip())
                lines.append("规则：提取核心元素，添加细节，构建合理场景。输出自然语言段落。")
            else:
                lines.append("Task: Generate from manual prompt only (no original).")
                lines.append("Manual: " + manual_text.strip())

        lines.append("")
        lines.append("现在请按格式输出：")
        lines.append("[POSITIVE] ")
        lines.append("[NEGATIVE] ")
        return "\n".join(lines)

# ---------- 视频提示词 ----------
def _format_video_prompt(manual_text: str, optional_text: str, mode: str, detail_level: str, output_lang: str) -> str:
    has_manual = bool(manual_text and manual_text.strip())
    has_optional = bool(optional_text and optional_text.strip())

    if has_optional and has_manual:
        task_type = "both"
    elif has_optional:
        task_type = "only_optional"
    else:
        task_type = "only_manual"

    # 根据语言分别设置描述字数阈值（英文提高以匹配中文语义量）
    if output_lang == "中文":
        if detail_level == "标准":
            min_keywords, max_keywords, min_desc_words = 15, 25, 200
        elif detail_level == "详细":
            min_keywords, max_keywords, min_desc_words = 25, 35, 400
        else:
            min_keywords, max_keywords, min_desc_words = 35, 45, 600
    else:  # 英文
        if detail_level == "标准":
            min_keywords, max_keywords, min_desc_words = 15, 25, 400   # 原200改为400
        elif detail_level == "详细":
            min_keywords, max_keywords, min_desc_words = 25, 35, 600   # 原400改为600
        else:
            min_keywords, max_keywords, min_desc_words = 35, 45, 800   # 原600改为800

    lines = []
    lines.append("【指令】直接输出正向和负向提示词及详细描述。严禁思考、分析、解释。")
    lines.append("严禁使用Markdown格式。")
    lines.append("")

    if output_lang == "中文":
        mode_text = "文生视频" if mode == "文生视频" else "图生视频"
        lines.append(f"你是一个专业的视频提示词专家。使用中文输出，{mode_text}模式。")
    else:
        mode_text = "text-to-video" if mode == "文生视频" else "image-to-video"
        lines.append(f"You are a professional video prompt expert. Use English output, {mode_text} mode.")
    lines.append("")

    # 核心规则（简化但保留关键）
    lines.append("【核心创作规则】")
    lines.append("1. 至少60%的正向关键词可来自通用元素，但必须保留核心角色和物体。")
    lines.append("2. 剧情描述（DESCRIPTION）是重点，必须充分扩展，达到字数要求。")
    lines.append("3. 合理预测下一步剧情：动作、环境变化、镜头运动。")
    lines.append("4. 为每个核心元素添加细节，主动扩展场景。")
    lines.append("")
    lines.append("【叙事逻辑】")
    lines.append("- 按时间顺序描述：首帧 → 镜头运动 → 动作序列 → 场景扩展。")
    lines.append("- 描述应达到字数要求。")
    lines.append("")

    # 输出格式
    if output_lang == "中文":
        lines.append("【输出格式】")
        lines.append(f"[POSITIVE]正向关键词（英文逗号分隔，{min_keywords}-{max_keywords}个，必须使用中文）")
        lines.append(f"[NEGATIVE]负向关键词（英文逗号分隔，至少{min_keywords//2}个，必须使用中文）")
        lines.append(f"[DESCRIPTION]详细场景描述（自然语言，必须使用中文，不少于{min_desc_words}字）")
    else:
        lines.append("【Output Format】")
        lines.append(f"[POSITIVE]positive keywords (comma separated, {min_keywords}-{max_keywords} words, English)")
        lines.append(f"[NEGATIVE]negative keywords (comma separated, at least {min_keywords//2} words, English)")
        lines.append(f"[DESCRIPTION]detailed scene description (natural language, English, at least {min_desc_words} words)")
    lines.append("")

    # 示例（简略）
    if output_lang == "中文":
        lines.append("【示例】")
        lines.append("用户输入：'一只金毛犬在黄昏的沙滩上，镜头跟随它'")
        lines.append("[POSITIVE]金毛犬, 沙滩, 黄昏, 逆光, 跟拍, 中景, 电影感, 慢动作, 写实")
        lines.append("[NEGATIVE]模糊, 抖动, 穿模, 动作卡顿, 低画质")
        lines.append("[DESCRIPTION]【首帧】中景，金毛犬站在潮湿沙滩上，夕阳金色轮廓光。【动作】先嗅沙子，突然加速奔跑，转向左侧，跃过浪花，冲向海鸥...")
    else:
        lines.append("【Example】")
        lines.append("User input: 'a golden retriever on a beach at dusk, camera following it'")
        lines.append("[POSITIVE]golden retriever, beach, dusk, backlighting, tracking shot, medium shot, cinematic, slow motion, realistic")
        lines.append("[NEGATIVE]blurry, shake, clipping, motion stutter, low quality")
        lines.append("[DESCRIPTION][First frame] Medium shot, golden retriever stands on wet sand, golden rim light...")
    lines.append("")

    # 任务输入（根据 task_type 添加详细规则）
    if task_type == "both":
        if output_lang == "中文":
            lines.append("【任务】综合以下内容创作。")
            lines.append("【原图/首帧描述】" + optional_text.strip())
            lines.append("【手工提示词】" + manual_text.strip())
            lines.append("规则：手工提示词优先替换冲突部分，不冲突元素保留融合。")
            lines.append(f"要求：正向关键词{min_keywords}-{max_keywords}个，描述不少于{min_desc_words}字。")
        else:
            lines.append("Task: Combine the following. Manual overrides conflicts; keep non-conflicting elements.")
            lines.append("Original/First frame: " + optional_text.strip())
            lines.append("Manual: " + manual_text.strip())
            lines.append(f"Requirement: positive keywords {min_keywords}-{max_keywords}, description at least {min_desc_words} words.")
    elif task_type == "only_optional":
        if output_lang == "中文":
            lines.append("【任务】基于原图/首帧描述生成视频剧情。")
            lines.append("【原图/首帧描述】" + optional_text.strip())
            lines.append("规则：提取核心角色、物体、场景，添加细节。合理预测后续动作、镜头运动、环境变化。按时间顺序描述。")
            lines.append(f"要求：正向关键词{min_keywords}-{max_keywords}个，描述不少于{min_desc_words}字。")
        else:
            lines.append("Task: Generate video storyline from original/first frame description only.")
            lines.append("Original/First frame: " + optional_text.strip())
            lines.append("Rules: Extract core characters, objects, scenes; add details; predict actions, camera movement, environment changes. Describe chronologically.")
            lines.append(f"Requirement: positive keywords {min_keywords}-{max_keywords}, description at least {min_desc_words} words.")
    else:  # only_manual
        if output_lang == "中文":
            lines.append("【任务】根据手工提示词构建完整视频剧情（无原图参考）。")
            lines.append("【手工提示词】" + manual_text.strip())
            lines.append("规则：提取核心元素，添加细节，设计动作序列、镜头运动、环境变化。构建合理场景。")
            lines.append(f"要求：正向关键词{min_keywords}-{max_keywords}个，描述不少于{min_desc_words}字。")
        else:
            lines.append("Task: Build complete video storyline from manual prompt only (no original).")
            lines.append("Manual: " + manual_text.strip())
            lines.append("Rules: Extract core elements; add details; design action sequences, camera movement, environment changes; build scene.")
            lines.append(f"Requirement: positive keywords {min_keywords}-{max_keywords}, description at least {min_desc_words} words.")

    lines.append("")
    lines.append("【严格输出格式】你必须输出三行，每行以方括号标签开头：")
    lines.append("[POSITIVE] ")
    lines.append("[NEGATIVE] ")
    lines.append("[DESCRIPTION] ")
    lines.append("")
    if output_lang == "中文":
        lines.append("现在直接输出（不要添加任何多余解释）：")
    else:
        lines.append("Now output directly (no extra explanation):")
    return "\n".join(lines)

# ---------- 内容反推 ----------
def _format_content_interrogation(image_base64: str, detail_level: str, output_lang: str) -> str:
    if output_lang == "中文":
        if detail_level == "标准":
            strategy = """详细描述图片中的主体（外观、姿态、服饰）、主要场景（环境元素）、整体光线方向、主色调和大致氛围。
要求：输出至少10个正向关键词，自然语言描述不少于120字。"""
        elif detail_level == "详细":
            strategy = """非常详细地描述图片的每一个视觉层面：
- 主体：具体特征（年龄、性别、表情、发型、动作、穿戴）
- 场景：精确环境（背景细节、前景元素、远近层次、空间关系）
- 光线：光源位置、强度、色温、阴影软硬、高光形态
- 色彩：主色、辅色、冷暖倾向、饱和度、对比度
- 质感：材质表现（皮肤、布料、金属、玻璃、水面等）
- 构图：镜头焦段、景别、视角、引导线、留白
- 情绪：整体氛围、情感倾向
要求：输出至少15个正向关键词，自然语言描述不少于250字。"""
        else:  # 极详细
            strategy = """极致深入地解剖图片的每一个视觉原子：
- 主体微观细节：瞳孔光斑、发丝走向、皮肤纹理、衣物褶皱与纤维、饰品反光
- 场景解剖：背景中每一类物体的名称、数量、相对位置、遮挡关系、远近虚实
- 光线深度：具体光源类型（日光灯/烛光/阴天天空/夕阳斜射）、光线衰减、二次反射、光晕、暗部补光
- 颜色层级：阴影色、中间调色、高光色，颜色间的过渡和冲突
- 材质科学：粗糙度、反射率、透射、次表面散射特征
- 构图数学：黄金分割点、对角线、框架内框架、视线流线
- 镜头光学：焦段具体数值（如24mm广角畸变、85mm人像压缩）、光圈导致的虚化量、光圈形状（星芒/圆形）
- 情绪微相：主体细微表情的肌肉运动、环境对情绪的烘托
要求：输出至少20个正向关键词，自然语言描述不少于500字。"""
        return f"""【指令】直接输出内容描述和负向提示词。严禁思考、分析、解释、Markdown。

你是一个专业的AI视觉内容分析专家。仔细观察图片，识别并描述所有内容。

【反推要求】{strategy}

【输出格式 - 必须严格遵守】
[POSITIVE]内容关键词（英文逗号分隔，数量符合上述要求，必须使用中文）
[NEGATIVE]负向提示词（英文逗号分隔，基于图片中不存在的常见瑕疵，至少8个词，必须使用中文）
[DESCRIPTION]自然语言描述（完整的句子，达到上述字数要求，中文）

【格式示例（极详细）】
[POSITIVE]年轻女性, 长发, 白色连衣裙, 沙滩, 海浪, 日落, 金色光, 逆光, 柔光, 中景, 低角度, 景深, 8K, 高细节, 电影感, 浪漫, 平静
[NEGATIVE]模糊, 低质量, 噪点, 畸变, 过曝, 抖动, 堵塞阴影, 色彩断层, 水印, 多余肢体, 不自然表情, 杂乱背景, 错误解剖, 重复纹理, 缺乏细节, 扁平光
[DESCRIPTION]这张图片是一幅优美的夕阳人像特写。一位年轻女性站在沙滩上，面朝大海，长发被海风吹起。她身穿白色连衣裙，裙摆微微飘动。夕阳位于画面右后方，产生强烈的逆光效果，在人物轮廓上形成金黄色的边缘光。光线温暖柔和，阴影拉长，整个场景笼罩在金色调中。中景低角度构图，前景有少量虚化的浪花，背景是波光粼粼的海面和淡紫色的晚霞。人物的皮肤质感细腻，带有微微的暖色高光。整体氛围浪漫、平静，像电影中的一帧画面。

【严格输出格式】你必须输出三行，每行以方括号标签开头：
[POSITIVE] 
[NEGATIVE] 
[DESCRIPTION] 

现在直接输出（不要添加任何多余解释）："""
    else:  # 英文
        if detail_level == "标准":
            strategy = """Describe in detail the subject (appearance, pose, clothing), main scene, overall lighting, dominant colors, and mood.
Requirements: at least 10 positive keywords, natural language description ≥120 words."""
        elif detail_level == "详细":
            strategy = """Describe every visual layer:
- Subject: specific features (age, gender, expression, hair, action, clothing)
- Scene: environment, background details, spatial relations
- Lighting: source, intensity, color temperature, shadow softness
- Color: main, secondary, saturation, contrast
- Texture: skin, fabric, metal, water
- Composition: focal length, shot size, angle, leading lines
- Mood: atmosphere, emotion
Requirements: at least 15 positive keywords, description ≥250 words."""
        else:
            strategy = """Dissect every visual atom:
- Subject micro-details: pupil specular, hair strands, skin pores, fabric folds
- Scene: exact objects, count, positions, occlusions
- Lighting: exact type (candlelight/overcast/golden hour), falloff, bounces
- Color: shadow/mid/highlight colors, transitions
- Material: roughness, reflectivity, subsurface scattering
- Composition: golden ratio, diagonals, frame-within-frame
- Lens optics: focal length, bokeh shape
- Emotion: micro-expressions, mood
Requirements: at least 20 positive keywords, description ≥500 words."""
        return f"""Instruction: Directly output content description and negative prompt. No thinking, no markdown.

You are a visual analysis expert. Observe the image carefully.

Analysis requirements: {strategy}

Strict output format:
[POSITIVE] (English keywords, comma separated, number as required)
[NEGATIVE] (English keywords, based on common flaws NOT in image, at least 8 words)
[DESCRIPTION] (natural language description, complete sentences, required length)

Example (extreme detail):
[POSITIVE] young woman, long hair, white dress, beach, waves, sunset, golden light, backlight, soft light, medium shot, low angle, depth of field, 8K, high detail, cinematic, romantic, calm
[NEGATIVE] blurry, low quality, noise, distortion, overexposed, shake, blocked shadows, color banding, watermark, text, extra limbs, unnatural expression, bad anatomy, repeating texture, color overflow, lack of detail, flat lighting
[DESCRIPTION] This image is a beautiful sunset portrait close-up. A young woman stands on the beach facing the sea, long hair blown by the sea breeze. She wears a white dress. The sun behind her creates a golden rim light. The scene is warm, golden, with soft shadows. Low-angle medium shot composition. The atmosphere is romantic and calm.

You MUST output exactly three lines, each starting with a bracket tag:
[POSITIVE] 
[NEGATIVE] 
[DESCRIPTION] 

Now output directly (no extra text):"""

# ========== 节点类 ==========

class AIConnector:
    @classmethod
    def INPUT_TYPES(cls):
        cfg = _CONFIG
        last = cfg.get("last_model", "") or ""
        return {
            "required": {
                "地址": ("STRING", {"default": cfg["base_url"], "placeholder": "http://127.0.0.1:1234"}),
                "令牌": ("STRING", {"default": cfg["token"], "placeholder": "API令牌（可选）"}),
                "模型": ("STRING", {"default": last, "placeholder": "输入模型名称"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("AI配置",)
    FUNCTION = "connect"
    CATEGORY = "AI"

    def connect(self, **kwargs):
        addr = kwargs["地址"]
        token = kwargs["令牌"]
        model = kwargs["模型"].strip()
        _save_config(base_url=addr, token=token, model=model)
        config_json = json.dumps({"address": addr, "token": token, "model": model})
        return (config_json,)

class AIContentInterrogator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图片": ("IMAGE",),
                "AI配置": ("STRING", {"forceInput": True}),
                "详细程度": (["标准", "详细", "极详细"], {"default": "详细"}),
                "输出语言": (["英文", "中文"], {"default": "中文"}),
            }
        }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("内容描述", "负向提示词")
    FUNCTION = "interrogate"
    CATEGORY = "AI"

    def interrogate(self, **kwargs):
        image_tensor = kwargs["图片"]
        config_json = kwargs["AI配置"]
        detail_level = kwargs["详细程度"]
        output_lang = kwargs["输出语言"]
        try:
            config = json.loads(config_json)
            addr = config.get("address", "")
            token = config.get("token", "")
            model = config.get("model", "")
        except:
            return ("", "")
        if not addr or not token or not model:
            return ("", "")
        try:
            if len(image_tensor.shape) == 4:
                img_tensor = image_tensor[0]
            else:
                img_tensor = image_tensor
            img = Image.fromarray((img_tensor.cpu().numpy() * 255).astype('uint8'))
            max_size = 2048
            if img.width > max_size or img.height > max_size:
                ratio = min(max_size / img.width, max_size / img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=95)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            req = _format_content_interrogation(img_base64, detail_level, output_lang)
            res = _ai_chat(addr, token, model, req, _CONFIG["timeout"], img_base64)
            res = res.strip().strip('"\'')
            parsed = _parse_marked_output(res)
            content_desc = ""
            if parsed["positive"] and parsed["description"]:
                content_desc = f"{parsed['positive']}\n\n{parsed['description']}"
            elif parsed["positive"]:
                content_desc = parsed["positive"]
            elif parsed["description"]:
                content_desc = parsed["description"]
            else:
                content_desc = res
            negative = parsed["negative"] if parsed["negative"] else "模糊, 低质量, 噪点, 畸变, 过曝"
            return (content_desc, negative)
        except Exception as e:
            return ("", "")

class AIImagePromptConverter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "AI配置": ("STRING", {"forceInput": True}),
                "内容描述": ("STRING", {"multiline": True, "placeholder": "输入内容描述（图片反推结果）"}),
                "手工提示词": ("STRING", {"multiline": True, "placeholder": "输入你的需求/意图（创作方向）"}),
                "生成模式": (["文生图", "图生图"], {"default": "文生图"}),
                "详细程度": (["标准", "详细", "极详细"], {"default": "详细"}),
                "输出语言": (["英文", "中文"], {"default": "英文"}),
                "输出类型": (["词汇", "整句"], {"default": "词汇"}),
            }
        }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("正向提示词", "负向提示词")
    FUNCTION = "convert_image"
    CATEGORY = "AI"

    def convert_image(self, **kwargs):
        config_json = kwargs["AI配置"]
        optional_prompt = kwargs["内容描述"].strip()
        manual_prompt = kwargs["手工提示词"].strip()
        mode = kwargs["生成模式"]
        detail_level = kwargs["详细程度"]
        output_lang = kwargs["输出语言"]
        output_type = kwargs["输出类型"]
        try:
            config = json.loads(config_json)
            addr = config.get("address", "")
            token = config.get("token", "")
            model = config.get("model", "")
        except:
            return ("AI配置格式错误", "AI配置格式错误")
        if not optional_prompt and not manual_prompt:
            return ("请填写内容描述或手工提示词", "请填写内容描述或手工提示词")
        if not addr or not token or not model:
            return ("请先配置AI连接器", "请先配置AI连接器")
        req = _format_image_prompt(manual_prompt, optional_prompt, mode, output_type, detail_level, output_lang)
        key = (manual_prompt, optional_prompt, model, mode, output_type, detail_level, output_lang)
        if key in _CACHE:
            cached = _CACHE[key]
            parsed = _parse_marked_output(cached)
            return (parsed["positive"], parsed["negative"])
        try:
            res = _ai_chat(addr, token, model, req, _CONFIG["timeout"])
            res = res.strip().strip('"\'')
            parsed = _parse_marked_output(res)
            if not parsed["positive"]:
                parsed["positive"] = "高质量图像"
            if not parsed["negative"]:
                parsed["negative"] = "模糊, 低质量, 噪点, 畸变"
            _CACHE[key] = f"[POSITIVE]{parsed['positive']}\n[NEGATIVE]{parsed['negative']}"
            return (parsed["positive"], parsed["negative"])
        except Exception as e:
            return (str(e), str(e))

class AIVideoPromptConverter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "AI配置": ("STRING", {"forceInput": True}),
                "内容描述": ("STRING", {"multiline": True, "placeholder": "输入内容描述（当前首帧图片描述）"}),
                "手工提示词": ("STRING", {"multiline": True, "placeholder": "输入创作意图"}),
                "生成模式": (["文生视频", "图生视频"], {"default": "图生视频"}),
                "详细程度": (["标准", "详细", "极详细"], {"default": "详细"}),
                "输出语言": (["中文", "英文"], {"default": "中文"}),
            }
        }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("正向提示词", "负向提示词")
    FUNCTION = "convert_video"
    CATEGORY = "AI"

    def convert_video(self, **kwargs):
        config_json = kwargs["AI配置"]
        optional_prompt = kwargs["内容描述"].strip()
        manual_prompt = kwargs["手工提示词"].strip()
        mode = kwargs["生成模式"]
        detail_level = kwargs["详细程度"]
        output_lang = kwargs["输出语言"]
        try:
            config = json.loads(config_json)
            addr = config.get("address", "")
            token = config.get("token", "")
            model = config.get("model", "")
        except:
            return ("AI配置格式错误", "AI配置格式错误")
        if not optional_prompt and not manual_prompt:
            return ("请填写内容描述或手工提示词", "请填写内容描述或手工提示词")
        if not addr or not token or not model:
            return ("请先配置AI连接器", "请先配置AI连接器")
        video_req = _format_video_prompt(manual_prompt, optional_prompt, mode, detail_level, output_lang)
        key = (manual_prompt, optional_prompt, model, mode, detail_level, output_lang)
        if key in _CACHE:
            cached = _CACHE[key]
            parsed = _parse_marked_output(cached)
            positive = parsed["positive"]
            if parsed["description"]:
                positive = f"{parsed['positive']}\n\n{parsed['description']}"
            return (positive, parsed["negative"])
        try:
            res = _ai_chat(addr, token, model, video_req, _CONFIG["timeout"])
            res = res.strip().strip('"\'')
            parsed = _parse_marked_output(res)
            positive = parsed["positive"] or "高质量视频"
            if parsed["description"]:
                positive = f"{positive}\n\n{parsed['description']}"
            negative = parsed["negative"] or "模糊, 抖动, 低画质"
            _CACHE[key] = f"[POSITIVE]{positive}\n[NEGATIVE]{negative}"
            return (positive, negative)
        except Exception as e:
            return (f"错误: {str(e)}", f"错误: {str(e)}")

NODE_CLASS_MAPPINGS = {
    "AIConnector": AIConnector,
    "AIContentInterrogator": AIContentInterrogator,
    "AIImagePromptConverter": AIImagePromptConverter,
    "AIVideoPromptConverter": AIVideoPromptConverter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIConnector": "🤖 AI 连接器",
    "AIContentInterrogator": "📝 AI 内容反推",
    "AIImagePromptConverter": "🎨 AI 图片提示词",
    "AIVideoPromptConverter": "🎬 AI 视频提示词",
}