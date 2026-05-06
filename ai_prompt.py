import re
import requests
import json
import os
import time
import base64
from typing import Dict, Tuple, List
from PIL import Image
import io

_CACHE: Dict[Tuple[str, str, str], str] = {}
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
    cleaned_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        
        if re.match(r'^(Thinking Process:|思考过程：|【思考】|\[思考\]|让我想想|我先|我需要|好的，|好，|明白了|理解了|收到|行，|嗯，|综上所述|总结|我认为|我觉得|分析：|思考：|推理：|解读：|Let me think|First|Then|Next|Finally|I think|Analysis:|Reasoning:)', line_stripped, re.IGNORECASE):
            continue
        
        cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines).strip()
    
    if not result:
        return text.strip()
    
    return result

def _parse_marked_output(text: str) -> dict:
    """解析带标记的输出格式，返回 {positive, negative, description}"""
    result = {
        "positive": "",
        "negative": "",
        "description": ""
    }
    
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
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": final_msg},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]
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
                    res = _clean_thinking_response(res)
                    return res
                elif "text" in choice:
                    res = choice["text"].strip()
                    res = _clean_thinking_response(res)
                    return res
                elif "content" in choice:
                    res = choice["content"].strip()
                    res = _clean_thinking_response(res)
                    return res
            
            if "output" in data and isinstance(data["output"], list):
                for item in data["output"]:
                    if item.get("type") == "message" and "content" in item:
                        res = item["content"].strip()
                        res = _clean_thinking_response(res)
                        return res
                for item in data["output"]:
                    if "content" in item:
                        res = item["content"].strip()
                        res = _clean_thinking_response(res)
                        return res
            
            if "response" in data:
                res = data["response"].strip()
                res = _clean_thinking_response(res)
                return res
            
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


def _format_image_prompt(manual_text: str, optional_text: str, mode: str, 
                         output_type: str, detail_level: str, output_lang: str) -> str:
    """格式化图片提示词，支持词汇模式(关键词)和整句模式(自然语言)"""
    has_manual = manual_text and manual_text.strip()
    has_optional = optional_text and optional_text.strip()
    
    # 确定任务类型
    if has_optional and has_manual:
        task_type = "both"
    elif has_optional:
        task_type = "only_optional"
    else:  # has_manual
        task_type = "only_manual"
    
    # 根据详细程度设置最低建议数量（非强制）
    if detail_level == "标准":
        if output_type == "词汇":
            min_positive = 60
            min_negative = 20
        else:
            min_positive_chars = 200
            min_negative_chars = 80
    elif detail_level == "详细":
        if output_type == "词汇":
            min_positive = 120
            min_negative = 30
        else:
            min_positive_chars = 400
            min_negative_chars = 150
    else:  # 极详细
        if output_type == "词汇":
            min_positive = 200
            min_negative = 40
        else:
            min_positive_chars = 600
            min_negative_chars = 200
    
    lines = []
    lines.append("【指令】直接输出正向和负向提示词，不要有任何思考过程、分析或解释。")
    lines.append("严禁使用任何Markdown格式。")
    lines.append("")
    
    # 角色描述（纯语言，避免中英文混用）
    if output_lang == "中文":
        if mode == "文生图":
            lines.append("你是一个专业的AI绘画提示词专家，使用中文输出，文生图模式。")
        else:
            lines.append("你是一个专业的AI绘画提示词专家，使用中文输出，图生图模式。")
    else:
        if mode == "文生图":
            lines.append("You are a professional AI image prompt expert. Use English output, text-to-image mode.")
        else:
            lines.append("You are a professional AI image prompt expert. Use English output, image-to-image mode.")
    lines.append("")
    
    # ========== 词汇模式 ==========
    if output_type == "词汇":
        # 公共规则（所有任务类型共用）
        if output_lang == "中文":
            lines.append("【词汇模式规则 - 中文关键词】")
            lines.append("1. 输出**中文关键词为主**，允许使用 8K、4K、UHD、HDR 等通用画质缩略词。")
            lines.append("   - 允许的英文缩略词：8K, 4K, UHD, HDR, PBR 等常见术语。")
            lines.append("   - 禁止的全英文单词：photorealistic, cinematic, sharp focus, volumetric light 等。这些必须用中文表达（照片级真实、电影感、清晰对焦、体积光）。")
            lines.append("2. 使用英文逗号加空格分隔关键词。")
            lines.append("3. 只输出关键词本身，严禁添加括号注释（如“（替换为主要互动对象）”）。")
            lines.append("4. 权重语法：使用英文半角括号 ( ) 和 [ ]，以及英文冒号 :。")
            lines.append("   正确：(杰作), (最佳质量:1.2), (玫瑰:1.1), [模糊背景], 8K, UHD")
            lines.append("   错误：（杰作）（中文括号）, 最佳质量：1.2（中文冒号）, photorealistic (英文单词)")
            lines.append("5. 语序建议：先写主体和核心动作，再写场景、构图、风格，最后写画质强化词。")
            lines.append("")
        else:
            lines.append("【Vocabulary Mode Rules - English Keywords】")
            lines.append("1. Output English keywords only, comma+space separated. No Chinese characters.")
            lines.append("2. Allowed short forms: 8K, 4K, UHD, HDR, PBR. Use weight syntax: ( ), [ ], :value.")
            lines.append("3. Word order: subject/action → scene → composition → style → quality.")
            lines.append("")
        
        # 根据任务类型添加差异化内容
        if task_type == "both":
            if output_lang == "中文":
                lines.append("【任务类型】综合原图描述与手工提示词（手工提示词优先修改冲突部分）")
                lines.append("【原图描述】" + optional_text.strip())
                lines.append("【手工提示词】" + manual_text.strip())
                lines.append("")
                lines.append("【处理规则】")
                lines.append("1. 原图描述提供基础画面（背景、主体、额外元素、光线、风格）。")
                lines.append("2. 如果手工提示词与原图中的动作、物体、场景发生冲突，则以手工提示词为准，替换冲突部分。")
                lines.append("3. 原图中不冲突的元素（如背景、光线、色调、额外人物/动物/物体）必须保留并融合。")
                lines.append("4. 为所有元素添加丰富细节（材质、光影、表情、环境）。")
                lines.append("5. 最后补充少量通用画质词（不超过20%）。")
                lines.append("")
                lines.append("【示例】")
                lines.append("原图描述：'一个穿蓝衬衫的男人坐在公园长椅上喝咖啡，旁边有一只松鼠。'")
                lines.append("手工提示词：'他在喂鸽子。'")
                lines.append("✅ 正确输出：蓝衬衫男人, 坐在公园长椅, 喂鸽子, 手心捧着面包屑, (鸽子:1.2), 白鸽子, 灰鸽子, 啄食, 松鼠, 绿树, 阳光, 中景, 写实风格, 8K")
            else:
                lines.append("【Task】Combine original and manual (manual overrides conflicts).")
                lines.append("Original: " + optional_text.strip())
                lines.append("Manual: " + manual_text.strip())
                lines.append("Rules: Keep non-conflicting elements; add details; add generic quality words ≤20%.")
                lines.append("Example: Original 'A man in blue shirt drinking coffee on a bench, with a squirrel.' Manual 'He is feeding pigeons.' → Good: man in blue shirt, sitting on bench, feeding pigeons, holding breadcrumbs, (pigeons:1.2), white pigeons, gray pigeons, pecking, squirrel, green trees, sunlight, medium shot, photorealistic, 8K")
        elif task_type == "only_optional":
            if output_lang == "中文":
                lines.append("【任务类型】仅基于原图描述生成提示词（无修改方向）")
                lines.append("【原图描述】" + optional_text.strip())
                lines.append("")
                lines.append("【处理规则】")
                lines.append("1. 从原图描述中提取所有视觉元素（主体、动作、场景、光线、风格等）。")
                lines.append("2. 为每个核心元素添加合理的细节扩展（材质、光影、氛围）。")
                lines.append("3. 根据场景添加协调的额外环境元素（例如森林可扩展松树、青苔）。")
                lines.append("4. 最后补充少量通用画质词（不超过20%）。")
                lines.append("")
                lines.append("【示例】")
                lines.append("原图描述：'一个穿红裙的女孩在花园里浇花，阳光明媚。'")
                lines.append("✅ 正确输出：红裙女孩, 花园浇花, 手持绿色水壶, 水珠洒落, 玫瑰花, 郁金香, 阳光, 金色光斑, 绿草地, 蝴蝶, 中景, 写实风格, 8K, UHD")
            else:
                lines.append("【Task】Generate from original description only.")
                lines.append("Original: " + optional_text.strip())
                lines.append("Rules: Extract all elements, add details, add scene expansions, then quality words.")
                lines.append("Example: Original 'A girl in red dress watering flowers in a sunny garden.' → Good: girl in red dress, watering flowers, holding green watering can, water droplets, roses, tulips, sunlight, golden flares, green grass, butterfly, medium shot, photorealistic, 8K")
        else:  # only_manual
            if output_lang == "中文":
                lines.append("【任务类型】仅基于手工提示词生成提示词（无原图参考）")
                lines.append("【手工提示词】" + manual_text.strip())
                lines.append("")
                lines.append("【处理规则】")
                lines.append("1. 从手工提示词中提取核心元素（主体、动作、物体）。")
                lines.append("2. 为每个核心元素添加丰富的细节扩展（材质、光影、表情）。")
                lines.append("3. 自行构建合理的场景、光线、构图、氛围来丰富画面。")
                lines.append("4. 最后补充少量通用画质词（不超过20%）。")
                lines.append("")
                lines.append("【示例】")
                lines.append("手工提示词：'一个老人坐在海边钓鱼。'")
                lines.append("✅ 正确输出：老人, 坐在礁石上, 手持鱼竿, 鱼线垂入海面, 海浪拍打, 夕阳, 金色逆光, 海鸥, 远处帆船, 低角度, 电影感, 8K")
            else:
                lines.append("【Task】Generate from manual prompt only (no original).")
                lines.append("Manual: " + manual_text.strip())
                lines.append("Rules: Extract core elements, add rich details, create reasonable scene/lighting/composition, then quality words.")
                lines.append("Example: Manual 'An old man sitting by the sea fishing.' → Good: old man, sitting on rock, holding fishing rod, fishing line dipping into sea, waves crashing, sunset, golden backlight, seagulls, distant sailboat, low angle, cinematic, 8K")
        
        # 共用数量建议和输出格式
        lines.append("")
        if output_lang == "中文":
            lines.append(f"【数量建议】正向关键词建议至少达到 {min_positive} 个词，负向至少 {min_negative} 个词。")
            lines.append("- 过少可能导致画面元素遗漏，请通过扩展细节自然增加词汇量，不要硬凑。")
            lines.append("")
            lines.append("【输出格式】")
            lines.append("[POSITIVE]（中文关键词+允许的缩略词，英文逗号+空格分隔，使用英文括号权重）")
            lines.append("[NEGATIVE]（中文关键词+允许的缩略词，英文逗号+空格分隔，使用英文括号权重）")
        else:
            lines.append(f"【Suggestion】Positive at least {min_positive} words, negative at least {min_negative} words.")
            lines.append("- Expand details naturally; avoid forcing words.")
            lines.append("")
            lines.append("【Output Format】")
            lines.append("[POSITIVE] (English keywords, comma+space, English parentheses)")
            lines.append("[NEGATIVE] (English keywords, comma+space, English parentheses)")
    
    # ========== 整句模式 ==========
    else:
        # 整句模式公共规则
        if output_lang == "中文":
            lines.append("【整句模式规则】输出自然语言段落，不要用逗号分隔的关键词。")
            lines.append("1. 使用中文，允许 8K、4K、UHD 等缩略词。")
            lines.append("2. 充分扩展细节（材质、光线、氛围等）。")
            lines.append("3. 负向提示词也用自然语言。")
            lines.append("")
        else:
            lines.append("【Sentence Mode Rules】Natural language paragraph, English only.")
            lines.append("1. Expand details (texture, lighting, atmosphere).")
            lines.append("2. Negative prompt also natural language.")
            lines.append("")
        
        # 根据任务类型添加差异化内容
        if task_type == "both":
            if output_lang == "中文":
                lines.append("【任务】综合原图与手工提示词（手工优先）。")
                lines.append("原图：" + optional_text.strip())
                lines.append("手工：" + manual_text.strip())
                lines.append("规则：冲突替换，不冲突保留，扩展细节。")
                lines.append("示例：原图'蓝衬衫男人喝咖啡，有松鼠'，手工'喂鸽子' → 正确输出：一个穿蓝衬衫的男人坐在公园长椅上，手里捧着面包屑，正在喂鸽子。旁边还有一只松鼠。背景是绿树和阳光。")
            else:
                lines.append("Task: Combine original and manual (manual overrides conflicts).")
                lines.append("Original: " + optional_text.strip())
                lines.append("Manual: " + manual_text.strip())
                lines.append("Example: Original 'A man in blue shirt drinking coffee, with a squirrel.' Manual 'feeding pigeons.' → Good: A man in blue shirt sits on a bench, holding breadcrumbs, feeding pigeons. A squirrel is nearby. Green trees, sunlight.")
        elif task_type == "only_optional":
            if output_lang == "中文":
                lines.append("【任务】基于原图描述生成。")
                lines.append("原图：" + optional_text.strip())
                lines.append("规则：提取元素，扩展细节，补充场景。")
                lines.append("示例：原图'红裙女孩在花园浇花' → 正确输出：一个穿红裙的女孩在花园里浇花，手持绿色水壶，水珠洒在玫瑰和郁金香上。阳光明媚，蝴蝶飞舞。中景，写实风格，8K超高清。")
            else:
                lines.append("Task: Generate from original description only.")
                lines.append("Original: " + optional_text.strip())
                lines.append("Example: Original 'A girl in red dress watering flowers in a sunny garden.' → Good: A girl in a red dress waters flowers in a garden, holding a green watering can. Water droplets fall on roses and tulips. Sunlight, butterflies. Medium shot, realistic, 8K.")
        else:  # only_manual
            if output_lang == "中文":
                lines.append("【任务】基于手工提示词生成（无原图）。")
                lines.append("手工：" + manual_text.strip())
                lines.append("规则：提取核心，扩展细节，构建场景。")
                lines.append("示例：手工'老人海边钓鱼' → 正确输出：一个老人坐在海边的礁石上，手持鱼竿，鱼线垂入海面。夕阳西下，金色逆光，海鸥飞翔，远处有帆船。低角度，电影感，8K。")
            else:
                lines.append("Task: Generate from manual prompt only (no original).")
                lines.append("Manual: " + manual_text.strip())
                lines.append("Example: Manual 'An old man sitting by the sea fishing.' → Good: An old man sits on a rock by the sea, holding a fishing rod. The line dips into the water. Sunset, golden backlight, seagulls, distant sailboat. Low angle, cinematic, 8K.")
        
        # 共用数量建议和输出格式
        lines.append("")
        if output_lang == "中文":
            lines.append(f"【数量建议】正向描述至少 {min_positive_chars} 字符，负向至少 {min_negative_chars} 字符。")
            lines.append("")
            lines.append("【输出格式】")
            lines.append("[POSITIVE]（自然语言段落，中文+允许的缩略词）")
            lines.append("[NEGATIVE]（自然语言段落，中文+允许的缩略词）")
        else:
            lines.append(f"【Suggestion】Positive at least {min_positive_chars} words, negative at least {min_negative_chars} words.")
            lines.append("")
            lines.append("【Output Format】")
            lines.append("[POSITIVE] (natural language paragraph, English only)")
            lines.append("[NEGATIVE] (natural language paragraph, English only)")
    
    lines.append("")
    lines.append("直接输出：")
    return "\n".join(lines)


def _format_video_prompt(manual_text: str, optional_text: str, mode: str, detail_level: str, output_lang: str) -> str:
    """格式化视频提示词 - 剧情扩展为主，关键词为辅"""
    has_manual = manual_text and manual_text.strip()
    has_optional = optional_text and optional_text.strip()
    
    # 确定任务类型（简化逻辑）
    if has_optional and has_manual:
        task_type = "both"
    elif has_optional:
        task_type = "only_optional"
    else:  # has_manual
        task_type = "only_manual"
    
    # 根据详细程度设定关键词数量和描述字数
    if detail_level == "标准":
        min_keywords = 15
        max_keywords = 25
        min_desc_words = 200
    elif detail_level == "详细":
        min_keywords = 25
        max_keywords = 35
        min_desc_words = 400
    else:  # 极详细
        min_keywords = 35
        max_keywords = 45
        min_desc_words = 600
    
    lines = []
    lines.append("【指令】直接输出正向和负向提示词及详细描述，不要有任何思考过程、分析或解释。")
    lines.append("严禁使用任何Markdown格式。")
    lines.append("")
    
    # 角色描述（纯语言，避免中英文混用）
    if output_lang == "中文":
        mode_text = "文生视频" if mode == "文生视频" else "图生视频"
        lines.append(f"你是一个专业的视频提示词专家。使用中文输出，{mode_text}模式。")
    else:
        mode_text = "text-to-video" if mode == "文生视频" else "image-to-video"
        lines.append(f"You are a professional video prompt expert. Use English output, {mode_text} mode.")
    lines.append("")
    
    # ========== 公共核心规则 ==========
    lines.append("【核心创作规则 - 必须严格遵守】")
    lines.append("1. **至少60%的正向关键词可以来自场景、光线、构图等通用元素**，但必须保留用户输入中的核心角色和物体。")
    lines.append("2. **剧情描述（DESCRIPTION）是重点**，必须充分扩展，达到字数要求。")
    lines.append("3. **视频可以而且应该合理预测下一步剧情**：根据当前画面，设计将要发生的动作、环境变化、镜头运动，只要逻辑合理即可。")
    lines.append("   - 例如：如果原图是一个人站在悬崖边，可以扩展“先转身，然后奔跑，最后纵身一跃”等后续动作。")
    lines.append("   - 可以添加合理的新角色、新物体、天气变化、时间流逝等，以推动剧情发展。")
    lines.append("4. 为每个核心元素添加丰富的细节，并主动扩展场景和环境。")
    lines.append("5. **禁止生搬硬套预定义类别**；直接输出关键词序列即可。")
    lines.append("")
    lines.append("【叙事逻辑要求】")
    lines.append("- 详细场景描述必须按时间顺序：先描述首帧（或起始状态），然后描述镜头运动，再按时间顺序描述动作序列（先...然后...接着...最后...）。")
    lines.append("- 动作之间要有明确的因果或连贯关系，避免动作堆叠。")
    lines.append("- 鼓励加入剧情转折、新事件、环境演变，使视频的每一秒都有新内容。")
    lines.append("- 描述应充分展开，达到上述字数要求。")
    lines.append("")
    
    # ========== 输出格式 ==========
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
    
    # ========== 示例（展示剧情扩展） ==========
    if output_lang == "中文":
        lines.append("【示例】")
        lines.append("用户输入：'一只金毛犬在黄昏的沙滩上，镜头跟随它'")
        lines.append("[POSITIVE]金毛犬, 沙滩, 黄昏, 逆光, 跟拍, 中景, 电影感, 慢动作, 写实")
        lines.append("[NEGATIVE]模糊, 抖动, 穿模, 动作卡顿, 低画质")
        lines.append("[DESCRIPTION]【首帧】中景，一只金毛犬站在潮湿的沙滩上，夕阳从背后照射，形成金色轮廓光。镜头固定，展示犬只全身。【镜头运动】镜头开始向前跟拍，保持与犬只的距离。【动作】犬只先低头嗅了嗅沙子，然后突然加速奔跑，四肢大幅度伸展。跑出几步后，它转向左侧，海浪冲上沙滩触碰到它的爪子，它兴奋地跃过浪花。紧接着，它向远处的一只海鸥冲去，海鸥受惊起飞，犬只追逐了几步后停下，回头看向镜头，吐着舌头。【场景扩展】背景中海面波光粼粼，远处有渔船，天空从橙色渐变为紫色。几分钟后夜幕降临，沙滩上亮起几盏路灯，犬只的身影在灯光下拉长，最后慢镜头中它平静地坐下，望向大海。")
    else:
        lines.append("【Example】")
        lines.append("User input: 'a golden retriever on a beach at dusk, camera following it'")
        lines.append("[POSITIVE]golden retriever, beach, dusk, backlighting, tracking shot, medium shot, cinematic, slow motion, realistic")
        lines.append("[NEGATIVE]blurry, shake, clipping, motion stutter, low quality")
        lines.append("[DESCRIPTION][First frame] Medium shot, a golden retriever stands on wet sand, backlit by the setting sun. Camera static. [Camera movement] Camera begins tracking forward. [Action] Dog sniffs the sand, then breaks into a sprint. After a few strides, it veers left, a wave washes over its paws, and it leaps over the foam. Then it charges toward a seagull, the gull takes flight, the dog chases then stops, looks back, panting. [Scene expansion] Ocean sparkles, a fishing boat visible, sky transitions from orange to purple. As night falls, street lamps light up on the beach, the dog's silhouette stretches, and in slow motion it sits down, gazing at the sea.")
    lines.append("")
    
    # ========== 可选知识库 ==========
    if output_lang == "中文":
        lines.append("【可选的参考知识库 - 仅当需要时使用】")
        lines.append("光源类型: 日光、人工光、月光、实用光、火光、荧光、阴天光、混合光、晴天光")
        lines.append("光线类型: 柔光、硬光、高对比度、侧光、底光、低对比度、边缘光、剪影、背光、逆光")
        lines.append("时间段: 白天、夜晚、日出、日落、黎明、黄昏、黄金时刻")
        lines.append("景别: 特写、近景、中景、中近景、全景、远景、广角")
        lines.append("构图: 中心构图、平衡构图、左侧构图、右侧构图、对称构图、短边构图")
        lines.append("镜头焦段: 广角、中焦距、长焦、鱼眼")
        lines.append("镜头运动: 固定、推进、拉远、上摇、下摇、左移、右移、手持、跟随、环绕")
        lines.append("角色情绪: 愤怒、恐惧、高兴、悲伤、惊讶、沉思、平静、焦虑")
        lines.append("视觉风格: 写实、电影感、纪录片、像素、3D、二次元、印象派、油画")
        lines.append("特效: 慢动作、动态模糊、镜头光晕、移轴、延时")
    else:
        lines.append("【Optional Knowledge Base - use only if needed】")
        lines.append("Light Source: daylight, artificial light, moonlight, practical light, firelight, fluorescent, overcast, mixed light, sunlight")
        lines.append("Lighting: soft lighting, hard lighting, high contrast, side lighting, underlighting, low contrast, rim lighting, silhouette, backlighting")
        lines.append("Time: daytime, night, sunrise, sunset, dawn, dusk, golden hour")
        lines.append("Shot Size: close-up, medium close-up, medium shot, medium wide shot, wide shot, extreme wide shot")
        lines.append("Composition: center composition, balanced composition, left-heavy, right-heavy, symmetrical, short-side")
        lines.append("Lens: wide-angle, medium lens, telephoto, fisheye")
        lines.append("Camera Movement: static, push-in, pull-back, pan, tilt, tracking, handheld, follow, orbit")
        lines.append("Emotion: angry, fearful, happy, sad, surprised, pensive, calm, anxious")
        lines.append("Style: photorealistic, cinematic, documentary, pixel, 3D, anime, impressionist, oil painting")
        lines.append("Effects: slow motion, motion blur, lens flare, tilt-shift, time-lapse")
    lines.append("")
    
    # ========== 根据任务类型添加用户输入和具体要求 ==========
    if task_type == "both":
        if output_lang == "中文":
            lines.append("【任务】综合以下内容，创作高质量的正向/负向关键词和详细场景描述。")
            lines.append("【原图/首帧描述】（必须保留并扩展）：")
            lines.append(optional_text.strip())
            lines.append("")
            lines.append("【手工提示词】（扩展/修改方向，必须融入并扩展）：")
            lines.append(manual_text.strip())
            lines.append("")
            lines.append("【详细要求】")
            lines.append("- 从上述描述中提取核心角色、物体、场景，并添加细节。")
            lines.append("- **重点在于DESCRIPTION**，要详细设计后续剧情，包括动作序列、镜头运动、环境变化、时间流逝、新元素加入等。")
            lines.append("- 正向关键词只需包含主要对象、场景、光线、构图、风格即可，不必过多扩展。")
            lines.append("- 负向关键词针对常见视频瑕疵。")
            lines.append(f"- 确保正向关键词数量在{min_keywords}-{max_keywords}之间，描述不少于{min_desc_words}字。")
            lines.append("- 如果手工提示词与原图描述存在冲突（例如动作、物体不同），以手工提示词为准，替换冲突部分，不冲突的元素保留。")
        else:
            lines.append("【Task】Combine the following to create positive/negative keywords and detailed scene description.")
            lines.append("【Original/First frame description】（must keep and expand）：")
            lines.append(optional_text.strip())
            lines.append("")
            lines.append("【Manual prompt】（direction for modification, must integrate and expand）：")
            lines.append(manual_text.strip())
            lines.append("")
            lines.append("【Requirements】")
            lines.append("- Extract core characters, objects, scenes and add details.")
            lines.append("- **Focus on DESCRIPTION**: design subsequent actions, camera movement, environmental changes, time passage, new elements.")
            lines.append("- Positive keywords only need main objects, scene, lighting, composition, style.")
            lines.append("- Negative keywords target common video flaws.")
            lines.append(f"- Ensure positive keywords between {min_keywords}-{max_keywords} words, description at least {min_desc_words} words.")
            lines.append("- If manual prompt conflicts with original (action, object, etc.), override with manual; keep non-conflicting elements.")
    elif task_type == "only_optional":
        if output_lang == "中文":
            lines.append("【任务】基于以下原图/首帧描述，创作高质量的正向/负向关键词和详细场景描述。")
            lines.append("【原图/首帧描述】：")
            lines.append(optional_text.strip())
            lines.append("")
            lines.append("【详细要求】")
            lines.append("- 提取核心元素，并在DESCRIPTION中扩展剧情。")
            lines.append("- 合理预测后续动作、镜头运动、环境演变。")
            lines.append(f"- 正向关键词{min_keywords}-{max_keywords}个，描述不少于{min_desc_words}字。")
        else:
            lines.append("【Task】Based on the original/first frame description, create positive/negative keywords and detailed scene description.")
            lines.append("【Original/First frame description】：")
            lines.append(optional_text.strip())
            lines.append("")
            lines.append("【Requirements】")
            lines.append("- Extract core elements, expand storyline in DESCRIPTION.")
            lines.append("- Reasonably predict subsequent actions, camera movement, environmental changes.")
            lines.append(f"- Positive keywords {min_keywords}-{max_keywords} words, description at least {min_desc_words} words.")
    else:  # only_manual
        if output_lang == "中文":
            lines.append("【任务】根据以下手工提示词，创作高质量的正向/负向关键词和详细场景描述（无原图参考）。")
            lines.append("【手工提示词】：")
            lines.append(manual_text.strip())
            lines.append("")
            lines.append("【详细要求】")
            lines.append("- 根据手工提示词构建完整视频剧情，DESCRIPTION是关键。")
            lines.append("- 合理设计动作序列、镜头运动、环境变化等。")
            lines.append(f"- 正向关键词{min_keywords}-{max_keywords}个，描述不少于{min_desc_words}字。")
        else:
            lines.append("【Task】Based on the manual prompt only (no original reference), create positive/negative keywords and detailed scene description.")
            lines.append("【Manual prompt】：")
            lines.append(manual_text.strip())
            lines.append("")
            lines.append("【Requirements】")
            lines.append("- Build complete video storyline from manual prompt; DESCRIPTION is key.")
            lines.append("- Reasonably design action sequences, camera movement, environmental changes.")
            lines.append(f"- Positive keywords {min_keywords}-{max_keywords} words, description at least {min_desc_words} words.")
    
    lines.append("")
    if output_lang == "中文":
        lines.append("直接输出：")
    else:
        lines.append("Direct output:")
    return "\n".join(lines)


def _format_content_interrogation(image_base64: str, detail_level: str, output_lang: str) -> str:
    """格式化内容反推提示词 - 输出内容描述和负向提示词（深层次描述），语言完全分离"""
    if output_lang == "中文":
        # 中文版本 ==========
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
        
        return f"""【指令】直接输出内容描述和负向提示词，不要有任何思考过程、分析或解释。
严禁使用任何Markdown格式。

你是一个专业的AI视觉内容分析专家。请仔细观察图片，识别并描述图片中的所有内容。

【反推要求】{strategy}

【输出格式 - 必须严格遵守】
[POSITIVE]内容关键词（用英文逗号分隔，数量符合上述要求，必须使用中文）
[NEGATIVE]负向提示词（用英文逗号分隔，必须使用中文，基于图片中不存在的常见问题或可能出现的瑕疵，数量不必与正向相等，但必须至少列出8个以上常见瑕疵词）
[DESCRIPTION]自然语言描述（用完整的句子描述，不要使用提示词格式，保持流畅的自然语言风格，必须达到上述字数要求）

【格式示例（中文极详细模式）】
[POSITIVE]年轻女性, 长发, 白色连衣裙, 沙滩, 海浪, 日落, 金色光, 逆光, 柔光, 中景, 低角度, 景深, 8K, 高细节, 电影感, 浪漫, 平静, 海风, 水光反射, 皮肤柔和, 纱裙飘逸
[NEGATIVE]模糊, 低质量, 噪点, 畸变, 过曝, 抖动, 堵塞阴影, 色彩断层, 水印, 文字, 多余肢体, 不自然表情, 杂乱背景, 错误解剖, 重复纹理, 颜色溢出, 缺乏细节, 扁平光
[DESCRIPTION]这张图片是一幅优美的夕阳人像特写。一位年轻女性站在沙滩上，面朝大海，长发被海风吹起。她身穿白色连衣裙，裙摆微微飘动。夕阳位于画面右后方，产生强烈的逆光效果，在人物轮廓上形成金黄色的边缘光。光线温暖柔和，阴影拉长，整个场景笼罩在金色调中。中景低角度构图，前景有少量虚化的浪花，背景是波光粼粼的海面和淡紫色的晚霞。人物的皮肤质感细腻，带有微微的暖色高光。整体氛围浪漫、平静，像电影中的一帧画面。

直接输出："""
    
    else:
        # 英文版本 ==========
        if detail_level == "标准":
            strategy = """Describe in detail the subject (appearance, pose, clothing), main scene (environmental elements), overall lighting direction, dominant colors, and general mood.
Requirements: Output at least 10 positive keywords, and natural language description of at least 120 words."""
        elif detail_level == "详细":
            strategy = """Describe every visual layer in great detail:
- Subject: specific features (age, gender, expression, hair, action, clothing)
- Scene: precise environment (background details, foreground elements, depth layering, spatial relations)
- Lighting: light source position, intensity, color temperature, shadow softness, highlight shape
- Color: main color, secondary color, cool/warm bias, saturation, contrast
- Texture: material representation (skin, fabric, metal, glass, water, etc.)
- Composition: lens focal length, shot size, angle, leading lines, negative space
- Mood: overall atmosphere, emotional tendency
Requirements: Output at least 15 positive keywords, and natural language description of at least 250 words."""
        else:  # 极详细
            strategy = """Dissect every visual atom of the image at an extreme depth:
- Subject micro-details: pupil specular, hair strand direction, skin pores, cloth wrinkles/fibers, jewelry reflections
- Scene anatomy: exact objects in background, their count, relative positions, occlusions, depth-of-field effects
- Lighting deep analysis: specific light source type (fluorescent/candlelight/overcast sky/golden hour), light falloff, secondary bounces, glow, fill light
- Color hierarchy: shadow tones, midtones, highlight colors, color transitions and conflicts
- Material science: roughness, reflectivity, transmission, subsurface scattering characteristics
- Composition mathematics: golden ratio points, diagonal lines, frame-within-frame, gaze flow
- Lens optics: exact focal length (e.g. 24mm wide distortion, 85mm portrait compression), amount of bokeh, aperture shape (star/circular)
- Emotional micro-expression: subtle muscle movements of the subject, environmental mood reinforcement
Requirements: Output at least 20 positive keywords, and natural language description of at least 500 words."""
        
        return f"""Instruction: Directly output content description and negative prompt. No thinking process, analysis, or explanation. No markdown.

You are a professional AI visual content analysis expert. Carefully observe the image and describe all content.

Analysis requirements: {strategy}

You MUST follow this exact output format:

[POSITIVE] (English keywords, comma separated, number as required above)
[NEGATIVE] (English keywords, comma separated, based on common flaws NOT present in the image. Must contain at least 8 words. Examples: blurry, low quality, noise, distortion, overexposed, shake, blocked shadows, color banding, watermark, text, extra limbs, unnatural expression, cluttered background, bad anatomy, repeating texture, color overflow, lack of detail, flat lighting)
[DESCRIPTION] (natural language description, complete sentences, must reach the required word count)

Example (extreme detail mode):
[POSITIVE] young woman, long hair, white dress, beach, waves, sunset, golden light, backlight, soft light, medium shot, low angle, depth of field, 8K, high detail, cinematic, romantic, calm, sea breeze, water reflection, soft skin, flowing dress
[NEGATIVE] blurry, low quality, noise, distortion, overexposed, shake, blocked shadows, color banding, watermark, text, extra limbs, unnatural expression, cluttered background, bad anatomy, repeating texture, color overflow, lack of detail, flat lighting
[DESCRIPTION] This image is a beautiful sunset portrait close-up. A young woman stands on the beach facing the sea, long hair blown by the sea breeze. She wears a white dress with the hem slightly fluttering. The sun is located at the upper right behind her, creating a strong backlight effect and a golden rim light around her silhouette. The light is warm and soft, with long shadows, and the whole scene is bathed in golden tones. Low-angle medium shot composition, with slightly blurred waves in the foreground, and a sparkling sea and pale purple twilight in the background. The skin texture is delicate with warm highlight reflections. The overall atmosphere is romantic, calm, like a frame from a movie.

Now output for the given image. Remember: [NEGATIVE] MUST contain at least 8 words. Do not skip it.

Direct output:"""


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
        config_json = json.dumps({
            "address": addr, 
            "token": token,
            "model": model
        })
        return (config_json,)


class AIContentInterrogator:
    """AI内容反推节点 - 输出内容描述和负向提示词"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图片": ("IMAGE",),
                "AI配置": ("STRING", {"forceInput": True, "placeholder": "连接AI连接器的输出"}),
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
            
            max_size = 1024
            if img.width > max_size or img.height > max_size:
                ratio = min(max_size / img.width, max_size / img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            req = _format_content_interrogation(img_base64, detail_level, output_lang)
            
            res = _ai_chat(addr, token, model, req, _CONFIG["timeout"], img_base64)
            res = res.strip().strip('"\'')
            
            res = re.sub(r'^(反推的提示词:|思考[：:]|分析[：:]|让我想想|好的|我理解了|明白了)\s*', '', res, flags=re.I)
            
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
            
            return (content_desc, parsed["negative"])
        except Exception as e:
            return ("", "")


class AIImagePromptConverter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "AI配置": ("STRING", {"forceInput": True, "placeholder": "连接AI连接器的输出"}),
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
            
            res = re.sub(r'^(转换后的提示词:|Converted prompt:|思考[：:]|分析[：:]|让我想想|好的|我理解了|明白了)\s*', '', res, flags=re.I)
            
            parsed = _parse_marked_output(res)
            
            _CACHE[key] = f"[POSITIVE]{parsed['positive']}\n[NEGATIVE]{parsed['negative']}"
            return (parsed["positive"], parsed["negative"])
        except Exception as e:
            return (str(e), str(e))


class AIVideoPromptConverter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "AI配置": ("STRING", {"forceInput": True, "placeholder": "连接AI连接器的输出"}),
                "内容描述": ("STRING", {"multiline": True, "placeholder": "输入内容描述（当前首帧图片描述）"}),
                "手工提示词": ("STRING", {"multiline": True, "placeholder": "输入创作意图（可包含上一段内容、首尾帧描述等）"}),
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
            return (parsed["positive"], parsed["negative"])
        
        try:
            res = _ai_chat(addr, token, model, video_req, _CONFIG["timeout"])
            res = res.strip().strip('"\'')
            
            res = re.sub(r'^(处理后的[^:]*:|优化后[^:]*:|视频提示词[^:]*:|思考[：:]|分析[：:]|让我想想|好的|我理解了|明白了)\s*', '', res, flags=re.I)
            
            parsed = _parse_marked_output(res)
            
            positive = ""
            if parsed["positive"] and parsed["description"]:
                positive = f"{parsed['positive']}\n\n{parsed['description']}"
            elif parsed["positive"]:
                positive = parsed["positive"]
            elif parsed["description"]:
                positive = parsed["description"]
            
            negative = parsed["negative"]
            
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