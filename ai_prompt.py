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


def _format_image_prompt(manual_text: str, optional_text: str, mode: str, detail_level: str, output_lang: str) -> str:
    """格式化图片提示词 - 保留具体内容并主动扩展细节"""
    has_manual = manual_text and manual_text.strip()
    has_optional = optional_text and optional_text.strip()
    
    if output_lang == "中文":
        lang_inst = "使用中文输出"
    else:
        lang_inst = "use English output"
    
    # 根据详细程度确定最少词数要求
    if detail_level == "标准":
        min_positive_words = 60
        min_negative_words = 20
    elif detail_level == "详细":
        min_positive_words = 120
        min_negative_words = 30
    else:  # 极详细
        min_positive_words = 200
        min_negative_words = 40
    
    lines = []
    lines.append("【指令】直接输出正向和负向提示词，不要有任何思考过程、分析或解释。")
    lines.append("严禁使用任何Markdown格式。")
    lines.append("")
    lines.append(f"你是一个专业的AI绘画提示词专家。{lang_inst}。")
    lines.append("")
    
    # 提示词公式参考
    if mode == "文生图":
        lines.append("【提示词公式参考】主体 + 场景 + 风格 + 光线 + 构图 + 质量")
    else:
        lines.append("【提示词公式参考】图生图：保留原图核心 + 描述修改方向")
    lines.append("")
    
    # 核心创作规则
    lines.append("【核心创作规则 - 必须严格遵守】")
    lines.append("1. **禁止只输出通用画质词**（如“杰作,最佳质量,8K,高细节”等）。通用词最多占正向提示词的20%。")
    lines.append("2. **至少80%的正向提示词必须来自用户输入的具体内容**，并且要**主动扩展细节**：")
    lines.append("   - 提取用户描述中的核心名词、形容词、动作。")
    lines.append("   - 为每个核心元素添加合理的细节（例如：“草帽” → “破旧的草帽, 帽檐磨损, 麻绳绑带”）。")
    lines.append("   - 扩展场景：根据已有环境添加协调的额外元素（例如：有“沙滩”可扩展“贝壳, 海浪泡沫, 湿沙反光”）。")
    lines.append("3. **不要生硬套用分类**：直接输出关键词序列即可。")
    lines.append("4. **顺序要求**：先列出具体内容词（核心元素及其细节），最后再添加少量通用质量/风格/光线词。")
    lines.append("")
    
    # 示例对比（略）
    if output_lang == "中文":
        lines.append("【正确与错误示例】")
        lines.append("用户输入：'一个戴着草帽的老渔夫坐在生锈的船上，手里拿着渔网，背景是黄昏的海港'")
        lines.append("✅ 正确输出（扩展充分）：草帽, 破旧草帽, 帽檐磨损, 老渔夫, 满脸皱纹, 深褐色皮肤, 白色胡茬, 粗糙双手, 指甲裂缝, 生锈的船, 剥落绿漆, 铆钉, 渔网, 麻绳, 木质甲板, 黄昏, 橙紫色天空, 长影子, 金色边缘光, 海港, 停泊船只, 码头木桩, 波光粼粼, 海鸥, 写实风格, 电影感, 柔光, 中景, 8K")
        lines.append("❌ 错误输出（只输出通用词）：杰作, 最佳质量, 8K, 高细节, 电影感, 柔光, 中景, 写实")
    else:
        lines.append("【Good vs Bad Examples】")
        lines.append("User input: 'an old fisherman in a straw hat sitting on a rusty boat, holding a fishing net, background is a dusk harbor'")
        lines.append("✅ Good: straw hat, worn straw hat, frayed brim, hemp strap, old fisherman, wrinkled face, dark brown skin, white stubble, rough hands, cracked nails, rusty boat, peeling green paint, rivets, fishing net, hemp rope, wooden deck, dusk, orange-purple sky, long shadows, golden rim light, harbor, moored boats, dock pilings, sparkling water, seagulls, realistic style, cinematic, soft lighting, medium shot, 8K")
        lines.append("❌ Bad: masterpiece, best quality, 8K, high detail, cinematic, soft lighting, medium shot, realistic")
    lines.append("")
    
    # 详细程度描述（与最少词数一致）
    lines.append(f"【详细程度】{detail_level}。正向提示词最少 {min_positive_words} 个词，负向提示词最少 {min_negative_words} 个词。")
    lines.append("关键词用英文逗号分隔，不要换行。")
    lines.append("")
    
    # 输出格式（动态显示数量要求）
    lines.append("【输出格式】")
    if output_lang == "中文":
        lines.append(f"[POSITIVE]正向提示词（必须使用中文，逗号分隔，至少 {min_positive_words} 个词）")
        lines.append(f"[NEGATIVE]负向提示词（必须使用中文，逗号分隔，至少 {min_negative_words} 个词）")
    else:
        lines.append(f"[POSITIVE]positive prompt (English, comma separated, at least {min_positive_words} words)")
        lines.append(f"[NEGATIVE]negative prompt (English, comma separated, at least {min_negative_words} words)")
    lines.append("")
    
    # 用户输入部分（保持不变）
    if has_optional and has_manual:
        lines.append("【任务】综合以下内容，创作高质量的正向和负向提示词。")
        lines.append("")
        lines.append("【原图描述】（核心内容基础，必须保留并扩展）：")
        lines.append(optional_text.strip())
        lines.append("")
        lines.append("【手工提示词】（扩展/修改方向，必须融入并扩展）：")
        lines.append(manual_text.strip())
        lines.append("")
        lines.append("【详细要求】")
        lines.append("1. 从上述两个描述中提取所有具体的视觉元素（人物、物体、场景、颜色、动作、材质等）。")
        lines.append("2. 为每个元素添加合理的细节扩展（参考上面的示例方式）。")
        lines.append("3. 根据场景添加协调的额外环境元素（例如：如果有“森林”，可以添加“松树、蕨类、青苔、光束”）。")
        lines.append("4. 如果手工提示词与原图描述有冲突，则按手工提示词调整，但仍尽量保留原图其他可用元素。")
        lines.append("5. 最后补充少量（不超过20%）通用画质/风格/光线/构图词。")
        lines.append("6. 负向提示词：根据扩展后的最终画面，排除常见的画质问题、结构问题和不需要的元素。")
        lines.append(f"7. **必须进行充分的细节扩展和场景扩展，确保正向提示词达到 {min_positive_words} 个词以上。**")
    elif has_optional and not has_manual:
        lines.append("【任务】基于以下原图描述，创作高质量的正向和负向提示词。")
        lines.append("")
        lines.append("【原图描述】：")
        lines.append(optional_text.strip())
        lines.append("")
        lines.append("【详细要求】")
        lines.append("- 提取所有核心元素并添加细节扩展。")
        lines.append("- 添加合理的场景扩展元素，使画面更丰富。")
        lines.append("- 最后补充少量通用画质/风格/光线/构图词。")
        lines.append(f"- **必须确保正向提示词达到 {min_positive_words} 个词以上。**")
    elif not has_optional and has_manual:
        lines.append("【任务】根据以下手工提示词，创作高质量的正向和负向提示词。")
        lines.append("")
        lines.append("【手工提示词】：")
        lines.append(manual_text.strip())
        lines.append("")
        lines.append("【详细要求】")
        lines.append("- 提取所有核心元素并添加细节扩展。")
        lines.append("- 添加合理的场景和氛围元素。")
        lines.append(f"- **必须确保正向提示词达到 {min_positive_words} 个词以上。**")
    
    lines.append("")
    lines.append(f"直接输出（{lang_inst}）：")
    
    return "\n".join(lines)


def _format_video_prompt(manual_text: str, optional_text: str, mode: str, detail_level: str, output_lang: str) -> str:
    """格式化视频提示词 - 基于文章方法论，强化叙事逻辑"""
    has_manual = manual_text and manual_text.strip()
    has_optional = optional_text and optional_text.strip()
    
    if output_lang == "中文":
        lang_inst = "使用中文输出"
        mode_text = "文生视频" if mode == "文生视频" else "图生视频"
        keyword_lang = "必须使用中文"
        
        # 基于文章的美学控制知识库
        knowledge = """【视觉元素知识库 - 按需选用】

光源类型: 日光、人工光、月光、实用光、火光、荧光、阴天光、混合光、晴天光
光线类型: 柔光、硬光、高对比度、侧光、底光、低对比度、边缘光、剪影、背光、逆光
时间段: 白天、夜晚、日出、日落、黎明、黄昏、黄金时刻
景别: 特写、近景、中景、中近景、全景、远景、广角
构图: 中心构图、平衡构图、左侧构图、右侧构图、对称构图、短边构图
镜头焦段: 广角、中焦距、长焦、鱼眼
镜头运动: 固定、推进、拉远、上摇、下摇、左移、右移、手持、跟随、环绕
角色情绪: 愤怒、恐惧、高兴、悲伤、惊讶、沉思、平静、焦虑
视觉风格: 写实、电影感、纪录片、像素、3D、二次元、印象派、油画
特效: 慢动作、动态模糊、镜头光晕、移轴、延时"""
    else:
        lang_inst = "use English output"
        mode_text = "text-to-video" if mode == "文生视频" else "image-to-video"
        keyword_lang = "must use English"
        
        knowledge = """【Visual Elements Knowledge Base】

Light Source: daylight, artificial light, moonlight, practical light, firelight, fluorescent, overcast, mixed light, sunlight
Lighting: soft lighting, hard lighting, high contrast, side lighting, underlighting, low contrast, rim lighting, silhouette, backlighting
Time: daytime, night, sunrise, sunset, dawn, dusk, golden hour
Shot Size: close-up, medium close-up, medium shot, medium wide shot, wide shot, extreme wide shot
Composition: center composition, balanced composition, left-heavy, right-heavy, symmetrical, short-side
Lens: wide-angle, medium lens, telephoto, fisheye
Camera Movement: static, push-in, pull-back, pan, tilt, tracking, handheld, follow, orbit
Emotion: angry, fearful, happy, sad, surprised, pensive, calm, anxious
Style: photorealistic, cinematic, documentary, pixel, 3D, anime, impressionist, oil painting
Effects: slow motion, motion blur, lens flare, tilt-shift, time-lapse"""
    
    lines = []
    lines.append("【指令】直接输出正向和负向提示词，不要有任何思考过程、分析或解释。")
    lines.append("严禁使用任何Markdown格式。")
    lines.append("")
    lines.append(f"你是一个专业的视频提示词专家。{lang_inst}，{mode_text}模式。")
    lines.append("")
    
    # 核心方法论（基于文章）
    lines.append("【核心方法论】")
    lines.append("提示词 = 关键词组合 + 详细场景描述")
    lines.append("")
    lines.append(f"关键词组合：从知识库中选择10-20个关键词，用英文逗号分隔，{keyword_lang}，服务于统一的艺术目标")
    lines.append("")
    lines.append("详细场景描述结构：")
    lines.append("【首帧】描述画面起始状态")
    lines.append("【镜头运动】描述镜头如何运动")
    lines.append("【动作】按时间顺序描述动作序列（用'先...然后...接着...最后...'连接）")
    lines.append("【场景扩展】描述环境变化和细节")
    lines.append("【转场】描述如何过渡到下一段（如需要）")
    lines.append("")
    
    # 叙事逻辑强化
    lines.append("【叙事逻辑规则 - 必须遵守】")
    lines.append("")
    lines.append("1. 时间顺序：动作必须按发生先后描述")
    lines.append("   - 使用：'先'、'然后'、'接着'、'紧接着'、'之后'、'最后'")
    lines.append("")
    lines.append("2. 因果关系：动作之间的因果必须明确")
    lines.append("   - 使用：'导致'、'使得'、'于是'、'从而'")
    lines.append("")
    lines.append("3. 动作密度：每5秒视频最多包含2-4个主要动作节点")
    lines.append("   - 每个动作节点用1-2句话独立描述")
    lines.append("   - 禁止将多个动作堆叠在一句话里")
    lines.append("")
    lines.append("4. 禁止的错误写法示例：")
    lines.append("   - ❌ '角色奔跑、跳跃、转身、落地挥手'（动作堆叠，无顺序）")
    lines.append("   - ✅ '角色先快速奔跑，然后纵身一跳，接着在空中转身，最后落地向大家挥手'")
    lines.append("")
    
    # 输出格式
    lines.append("【输出格式】")
    lines.append(f"[POSITIVE]关键词（英文逗号分隔，10-20个，{keyword_lang}）")
    lines.append(f"[NEGATIVE]负向关键词（英文逗号分隔，8-15个，{keyword_lang}）")
    lines.append(f"[DESCRIPTION]详细场景描述（按上述结构组织，必须使用{output_lang}）")
    lines.append("")
    lines.append("【格式示例】")
    if output_lang == "中文":
        lines.append("[POSITIVE]中景, 黄金时刻, 逆光, 暖色调, 跟拍, 柔光, 侧光, 动态模糊, 慢动作, 电影感, 宁静")
        lines.append("[NEGATIVE]模糊, 低质量, 畸形的手, 多余的手指, 水印, 文字, 抖动, 穿模, 动作卡顿")
        lines.append("[DESCRIPTION]【首帧】中景，一个穿白裙的女孩站在沙滩上面朝大海。【镜头运动】镜头向后拉远，从近景变为全景。【动作】女孩先开始奔跑，然后镜头切换到跟拍，跟随她的侧脸移动。【场景扩展】随着她的奔跑，镜头慢慢拉远，展现更广阔的海滩：金色沙滩延伸到远方，左边是礁石群，右边是椰林，海浪一层层拍打。【转场】淡入淡出，暗示时间流逝。")
    else:
        lines.append("[POSITIVE]medium shot, golden hour, backlighting, warm colors, tracking shot, soft lighting, side lighting, motion blur, slow motion, cinematic, serene")
        lines.append("[NEGATIVE]blurry, low quality, bad anatomy, extra fingers, watermark, text, shake, clipping, motion stutter")
        lines.append("[DESCRIPTION][First Frame] Medium shot, a girl in a white dress stands on the beach facing the sea. [Camera Movement] The camera pulls back, transitioning from medium to wide shot. [Action] First the girl starts running, then the camera switches to tracking shot, following her profile. [Scene Expansion] As she runs, the camera slowly pulls back to reveal the vast beach: golden sand stretching into the distance, reefs on the left, palm trees on the right, waves lapping the shore. [Transition] Fade to black, suggesting time passing.")
    lines.append("")
    
    lines.append(knowledge)
    lines.append("")
    
    # 用户输入
    if has_optional and has_manual:
        lines.append("【原图描述】")
        lines.append(optional_text.strip())
        lines.append("")
        lines.append("【手工提示词】")
        lines.append(manual_text.strip())
    elif has_optional and not has_manual:
        lines.append("【原图描述】")
        lines.append(optional_text.strip())
    elif not has_optional and has_manual:
        lines.append("【手工提示词】")
        lines.append(manual_text.strip())
    
    lines.append("")
    lines.append(f"直接输出（{lang_inst}）：")
    
    return "\n".join(lines)


def _format_content_interrogation(image_base64: str, detail_level: str, output_lang: str) -> str:
    """格式化内容反推提示词 - 输出内容描述和负向提示词（深层次描述）"""
    if output_lang == "中文":
        lang_inst = "使用中文输出"
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
    else:
        lang_inst = "use English output"
        if detail_level == "standard":
            strategy = """Describe in detail the subject (appearance, pose, clothing), main scene (environmental elements), overall lighting direction, dominant colors, and general mood.
Requirements: Output at least 10 positive keywords, and natural language description of at least 120 words."""
        elif detail_level == "detailed":
            strategy = """Describe every visual layer in great detail:
- Subject: specific features (age, gender, expression, hair, action, clothing)
- Scene: precise environment (background details, foreground elements, depth layering, spatial relations)
- Lighting: light source position, intensity, color temperature, shadow softness, highlight shape
- Color: main color, secondary color, cool/warm bias, saturation, contrast
- Texture: material representation (skin, fabric, metal, glass, water, etc.)
- Composition: lens focal length, shot size, angle, leading lines, negative space
- Mood: overall atmosphere, emotional tendency
Requirements: Output at least 15 positive keywords, and natural language description of at least 250 words."""
        else:  # extreme
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
    
    return f"""【指令】直接输出内容描述和负向提示词，不要有任何思考过程、分析或解释。
严禁使用任何Markdown格式。

你是一个专业的AI视觉内容分析专家。请仔细观察图片，识别并描述图片中的所有内容。

【反推要求】{strategy}

【输出格式 - 必须严格遵守】
[POSITIVE]内容关键词（用英文逗号分隔，数量符合上述要求，必须使用{output_lang}）
[NEGATIVE]负向提示词（用英文逗号分隔，数量至少与正向关键词相等，必须使用{output_lang}，基于图片中不存在的常见问题或可能出现的瑕疵）
[DESCRIPTION]自然语言描述（用完整的句子描述，不要使用提示词格式，保持流畅的自然语言风格，必须达到上述字数要求）

【格式示例（中文极详细模式）】
[POSITIVE]年轻女性, 长发, 白色连衣裙, 沙滩, 海浪, 日落, 金色光, 逆光, 柔光, 中景, 低角度, 景深, 8K, 高细节, 电影感, 浪漫, 平静, 海风, 水光反射, 皮肤柔和, 纱裙飘逸
[NEGATIVE]模糊, 低质量, 噪点, 畸变, 过曝, 抖动, 堵塞阴影, 色彩断层, 水印, 文字, 多余肢体, 不自然表情, 杂乱背景, 错误解剖, 重复纹理, 颜色溢出, 缺乏细节, 扁平光
[DESCRIPTION]这张图片是一幅优美的夕阳人像特写。一位年轻女性站在沙滩上，面朝大海，长发被海风吹起。她身穿白色连衣裙，裙摆微微飘动。夕阳位于画面右后方，产生强烈的逆光效果，在人物轮廓上形成金黄色的边缘光。光线温暖柔和，阴影拉长，整个场景笼罩在金色调中。中景低角度构图，前景有少量虚化的浪花，背景是波光粼粼的海面和淡紫色的晚霞。人物的皮肤质感细腻，带有微微的暖色高光。整体氛围浪漫、平静，像电影中的一帧画面。

直接输出："""


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
        
        req = _format_image_prompt(manual_prompt, optional_prompt, mode, detail_level, output_lang)
        key = (manual_prompt, optional_prompt, model, mode, detail_level, output_lang)
        
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