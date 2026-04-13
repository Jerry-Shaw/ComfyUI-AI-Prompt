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
    """清理响应中的思考过程，只保留最终结果 - 保守版本"""
    if not text:
        return text
    
    # 检查是否包含思考标签（只有明确的行首标签才删除）
    lines = text.split('\n')
    cleaned_lines = []
    skip_current_line = False
    
    for line in lines:
        line_stripped = line.strip()
        
        # 只删除明确的思考标签行（这些行通常是独立的）
        if re.match(r'^(Thinking Process:|思考过程：|【思考】|\[思考\]|让我想想|我先|我需要|好的，|好，|明白了|理解了|收到|行，|嗯，|首先|第一步|然后|接着|接下来|之后|最后|综上所述|总结|我认为|我觉得|分析：|思考：|推理：|解读：|Let me think|First|Then|Next|Finally|I think|Analysis:|Reasoning:)\s*$', line_stripped, re.IGNORECASE):
            skip_current_line = True
            continue
        
        # 删除单独的行首标签后跟空内容的情况
        if re.match(r'^(Thinking Process:|思考过程：|【思考】|\[思考\])\s*$', line_stripped, re.IGNORECASE):
            continue
        
        # 如果上一行是思考标签，跳过空行
        if skip_current_line and not line_stripped:
            skip_current_line = False
            continue
        
        skip_current_line = False
        cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines).strip()
    
    # 如果清理后为空，返回原文本（但去掉首尾空白）
    if not result:
        return text.strip()
    
    return result

def _ai_chat(url: str, token: str, model: str, msg: str, timeout: int = 60, image_base64: str = None) -> str:
    """通用的AI聊天接口，支持OpenAI兼容的API，支持多模态"""
    base = url.rstrip('/')
    
    # 在消息末尾加上 /no_think（Qwen/MiMo 支持，LM Studio 会当作普通文本，不会报错）
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
    
    # 纯文本请求
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

def _format_image_prompt(manual_text: str, optional_text: str, mode: str, detail_level: str, prompt_type: str, output_lang: str) -> str:
    """格式化图片提示词 - 支持语义融合模式，忠于原图，支持场景扩展"""
    header = "【绝对指令】直接输出最终结果，严禁任何思考过程、分析或解释。"
    
    thinking_ban = """
【严重警告】
- 绝对不允许输出任何思考过程、分析或解释
- 直接输出最终提示词，不要有任何前缀、后缀或标签
- 不要输出"好的"、"我理解了"、"根据要求"等开场白
- 只输出提示词本身
"""
    
    has_manual = manual_text and manual_text.strip()
    has_optional = optional_text and optional_text.strip()
    
    input_section_parts = []
    
    if has_manual and has_optional:
        # ========== 语义融合模式：理解手工提示词的意图，融合到原图描述中，支持场景扩展 ==========
        input_section_parts.append("【任务说明 - 语义融合模式（忠于原图 + 场景扩展）】")
        input_section_parts.append("")
        input_section_parts.append("【核心原则】")
        input_section_parts.append("1. 【忠于原图】- 最终画面必须与原图描述的核心视觉元素保持一致")
        input_section_parts.append("2. 【场景扩展】- 根据手工提示词的意图，可以合理扩展原图之外的场景内容")
        input_section_parts.append("3. 【风格统一】- 扩展的内容必须与原图的风格、世界观、光影保持一致")
        input_section_parts.append("")
        input_section_parts.append("【原图描述】（必须以此为基础，保持其核心内容）：")
        input_section_parts.append(optional_text.strip())
        input_section_parts.append("")
        input_section_parts.append("【手工提示词】（理解意图，在原图基础上扩展场景）：")
        input_section_parts.append(manual_text.strip())
        input_section_parts.append("")
        input_section_parts.append("【场景扩展规则】")
        input_section_parts.append("1. 原图描述的画面是【核心】和【参考】，不是全部内容")
        input_section_parts.append("2. 可以根据手工提示词的意图，合理想象原图之外的空间")
        input_section_parts.append("3. 扩展的内容必须与原图风格、世界观保持一致")
        input_section_parts.append("4. 扩展可以是：")
        input_section_parts.append("   - 空间延伸：展现更广阔的环境（背景延伸、周边环境）")
        input_section_parts.append("   - 时间推进：表现时间变化（日出到日落、季节变化）")
        input_section_parts.append("   - 情节发展：添加新元素或人物互动")
        input_section_parts.append("   - 细节丰富：补充原图中隐含但未描述的细节")
        input_section_parts.append("5. 扩展不能完全替换原图的核心元素")
        input_section_parts.append("")
        input_section_parts.append("【图片构图指导】")
        input_section_parts.append("根据场景需要，可以设计：")
        input_section_parts.append("- 景别：特写、近景、中景、全景、远景")
        input_section_parts.append("- 角度：平视、俯视、仰视、鸟瞰、低角度")
        input_section_parts.append("- 构图：中心构图、三分法、引导线、框架构图、对称构图")
        input_section_parts.append("- 光线：顺光、侧光、逆光、轮廓光、柔光、硬光")
        input_section_parts.append("- 色调：暖色调、冷色调、对比色、单色调")
        input_section_parts.append("")
        input_section_parts.append("【融合规则 - 重要】")
        input_section_parts.append("1. 【忠于原图】是最高优先级，原图的核心场景、主体、构图必须保留")
        input_section_parts.append("2. 理解手工提示词的语义意图，但只能【在原图基础上】进行扩展或修改")
        input_section_parts.append("3. 禁止完全替换原图的主体或场景，只能做局部调整和扩展")
        input_section_parts.append("4. 手工提示词可能包含：")
        input_section_parts.append("   - 添加元素（如：'加一只猫'）→ 在原图中添加该元素")
        input_section_parts.append("   - 修改属性（如：'把红色衣服改成蓝色'）→ 修改指定属性")
        input_section_parts.append("   - 氛围调整（如：'更阴郁的氛围'）→ 调整光线/色调，不改变主体")
        input_section_parts.append("   - 动作/状态（如：'让角色奔跑起来'）→ 修改角色的动作")
        input_section_parts.append("   - 场景扩展（如：'展现周围的森林'）→ 在原图基础上扩展环境")
        input_section_parts.append("5. 如果手工提示词要求完全改变场景或主体，优先保留原图，只做最小必要修改")
        input_section_parts.append("6. 输出描述必须与原图有清晰的可识别连续性")
        input_section_parts.append("")
        input_section_parts.append("【正确示例 - 忠于原图 + 场景扩展】")
        input_section_parts.append("原图描述: '一个穿红裙的女孩站在沙滩上，阳光明媚，海浪在背景中拍打'")
        input_section_parts.append("手工提示词: '让她跑起来，展现海滩的广阔'")
        input_section_parts.append("正确输出: '一个穿红裙的女孩在广阔的沙滩上奔跑，阳光明媚，海浪在背景中拍打，裙摆随风飘动。沙滩延伸到远方，左边是礁石群，右边是椰林，天空飘着几朵白云。'")
        input_section_parts.append("（✅ 保留了原图的主体、场景、光线，扩展了海滩环境，修改了动作）")
        input_section_parts.append("")
        input_section_parts.append("【错误示例 - 偏差太大，禁止这样做】")
        input_section_parts.append("原图描述: '一个穿红裙的女孩站在沙滩上，阳光明媚'")
        input_section_parts.append("手工提示词: '让她跑起来，展现海滩的广阔'")
        input_section_parts.append("错误输出: '一个运动员在田径场上奔跑，汗流浃背'")
        input_section_parts.append("（❌ 完全改变了场景和主体，与原图无关，禁止！）")
        input_section_parts.append("")
        input_section_parts.append("【融合示例 - 复杂意图的场景扩展】")
        input_section_parts.append("原图描述: '一片宁静的森林，阳光透过树叶洒下斑驳光影，小溪潺潺流过'")
        input_section_parts.append("手工提示词: '这是奇幻冒险的风格，需要主角快速前进，不要停留在当前画面'")
        input_section_parts.append("正确理解: 在保持森林场景的基础上，添加奇幻冒险风格的元素，扩展森林的更多细节")
        input_section_parts.append("正确输出: '一片神秘的森林中，阳光透过茂密的树叶洒下斑驳光影，小溪潺潺流过。一个背着行囊的冒险者快步走在溪边，踏过布满青苔的石头。远处可见参天古树，藤蔓缠绕，树根盘错，偶尔有萤火虫般的光芒在林间飘动。冒险者拨开树枝，前方出现一座古老的石桥，桥下溪水清澈见底，桥的另一端通向迷雾笼罩的深处。'")
        input_section_parts.append("（✅ 保留了原图的森林、阳光、小溪，扩展了森林的更多元素，添加了冒险者）")
        
    elif has_manual and not has_optional:
        # ========== 只有手工提示词 ==========
        input_section_parts.append("【任务说明 - 从零生成模式】")
        input_section_parts.append("")
        input_section_parts.append("请根据手工提示词，直接生成一个高质量的画面描述。")
        input_section_parts.append("")
        input_section_parts.append("【手工提示词】：")
        input_section_parts.append(manual_text.strip())
        
    elif not has_manual and has_optional:
        # ========== 只有原图描述 ==========
        input_section_parts.append("【任务说明 - 优化模式】")
        input_section_parts.append("")
        input_section_parts.append("请基于原图描述，优化成一个更高质量的画面描述，但不要改变核心内容。")
        input_section_parts.append("")
        input_section_parts.append("【原图描述】：")
        input_section_parts.append(optional_text.strip())
    
    if not input_section_parts:
        return None
    
    input_section = "\n".join(input_section_parts)
    
    # 详细程度策略
    if detail_level == "标准":
        strategy = """【详细程度：标准】
- 简洁明了，突出重点
- 只描述核心元素"""
    elif detail_level == "详细":
        strategy = """【详细程度：详细】
- 适当补充环境、光线、质感
- 描述更加丰富"""
    else:
        strategy = """【详细程度：极详细】
- 电影级详细描述
- 包含光影、色彩、构图、氛围
- 尽可能丰富"""
    
    if output_lang == "中文":
        lang_instruction = "必须使用中文输出"
    else:
        lang_instruction = "必须使用英文输出"
    
    return f"""{header}
{thinking_ban}

你是一个专业的AI绘画提示词专家。

{strategy}

{input_section}

【输出要求】
1. 直接输出提示词，不要有任何其他内容
2. 不要输出思考过程、分析、解释
3. 【最高优先级】忠于原图描述，保持原图的核心视觉元素不变
4. 【场景扩展】可以合理扩展原图之外的内容，但要保持风格一致
5. {lang_instruction}

直接输出提示词："""

def _format_video_prompt(manual_text: str, optional_text: str, mode: str, detail_level: str, output_lang: str) -> str:
    """格式化视频提示词 - 支持语义融合模式，忠于原图，支持场景扩展、镜头语言和转场效果"""
    header = "【绝对指令】直接输出最终结果，严禁任何思考过程、分析或解释。"
    
    thinking_ban = """
【严重警告】
- 绝对不允许输出任何思考过程、分析或解释
- 直接输出最终提示词，不要有任何前缀、后缀或标签
- 不要输出"好的"、"我理解了"、"根据要求"等开场白
- 只输出提示词本身
"""
    
    has_manual = manual_text and manual_text.strip()
    has_optional = optional_text and optional_text.strip()
    
    input_section_parts = []
    
    if has_manual and has_optional:
        # ========== 语义融合模式：理解手工提示词的意图，融合到原图描述中，支持场景扩展、镜头语言和转场效果 ==========
        input_section_parts.append("【任务说明 - 语义融合模式（忠于原图 + 场景扩展 + 镜头语言 + 转场效果）】")
        input_section_parts.append("")
        input_section_parts.append("【核心原则】")
        input_section_parts.append("1. 【忠于原图】- 视频首帧必须与原图描述的核心视觉元素一致")
        input_section_parts.append("2. 【场景扩展】- 根据手工提示词的意图，可以合理扩展原图之外的场景内容")
        input_section_parts.append("3. 【镜头语言】- 根据剧情需要，可以设计镜头运动、景别变化、视角切换")
        input_section_parts.append("4. 【转场效果】- 根据剧情节奏和情感需要，可以设计各种转场效果")
        input_section_parts.append("5. 【风格统一】- 扩展的内容必须与原图的风格、世界观、光影保持一致")
        input_section_parts.append("")
        input_section_parts.append("【原图描述】（视频首帧必须以此为基础，保持其核心内容）：")
        input_section_parts.append(optional_text.strip())
        input_section_parts.append("")
        input_section_parts.append("【手工提示词】（理解意图，在原图基础上扩展场景、设计镜头和转场）：")
        input_section_parts.append(manual_text.strip())
        input_section_parts.append("")
        input_section_parts.append("【场景扩展规则】")
        input_section_parts.append("1. 原图描述的画面是【起点】和【参考】，不是全部内容")
        input_section_parts.append("2. 可以根据手工提示词的意图，合理想象原图之外的空间")
        input_section_parts.append("3. 扩展的内容必须与原图风格、世界观保持一致")
        input_section_parts.append("4. 扩展可以是：")
        input_section_parts.append("   - 空间延伸（镜头拉远/摇移，展现更广阔的环境）")
        input_section_parts.append("   - 时间推进（日出到日落，天气变化）")
        input_section_parts.append("   - 情节发展（人物移动、新角色出现、事件发生）")
        input_section_parts.append("   - 视角切换（从A角色切换到B角色的视角）")
        input_section_parts.append("   - 细节丰富（补充原图中隐含但未描述的细节）")
        input_section_parts.append("")
        input_section_parts.append("【镜头语言指导】")
        input_section_parts.append("根据手工提示词的意图和剧情需要，可以设计以下镜头：")
        input_section_parts.append("")
        input_section_parts.append("【景别变化】")
        input_section_parts.append("- 特写（Close-up）：聚焦面部表情或细节，强化情感")
        input_section_parts.append("- 近景（Medium Close-up）：胸部以上，展现表情和部分环境")
        input_section_parts.append("- 中景（Medium Shot）：腰部以上，展现动作和互动")
        input_section_parts.append("- 全景（Full Shot）：全身入画，展现完整动作")
        input_section_parts.append("- 远景（Long Shot）：人物在环境中较小，强调环境")
        input_section_parts.append("- 极远景（Extreme Long Shot）：人物几乎看不见，强调宏大场景")
        input_section_parts.append("")
        input_section_parts.append("【镜头运动】")
        input_section_parts.append("- 推镜头（Push In / Zoom In）：聚焦细节，强调重点，营造紧张感")
        input_section_parts.append("- 拉镜头（Pull Out / Zoom Out）：展现环境，交代空间关系，揭示更大场景")
        input_section_parts.append("- 摇镜头（Pan）：水平扫视，展现场景全貌，跟随运动方向")
        input_section_parts.append("- 移镜头（Truck）：平行移动，跟随主体，保持主体在画面中")
        input_section_parts.append("- 跟镜头（Follow）：追随主体运动，让观众代入，增强沉浸感")
        input_section_parts.append("- 升降镜头（Crane Up/Down）：垂直运动，展现空间层次，揭示布局")
        input_section_parts.append("- 环绕镜头（Orbit）：围绕主体旋转，全方位展示，增加戏剧性")
        input_section_parts.append("- 手持镜头（Handheld）：轻微晃动，增强真实感和紧迫感")
        input_section_parts.append("")
        input_section_parts.append("【视角选择】")
        input_section_parts.append("- 主观视角（POV）：从角色眼中看世界，强代入感")
        input_section_parts.append("- 客观视角（Third Person）：旁观者视角，叙事性")
        input_section_parts.append("- 越肩视角（Over-the-shoulder）：对话场景，展现互动关系")
        input_section_parts.append("- 鸟瞰视角（Bird's-eye view）：正上方俯视，展现全貌和布局")
        input_section_parts.append("- 低角度（Low Angle）：从下往上，强调力量和威严")
        input_section_parts.append("- 高角度（High Angle）：从上往下，强调脆弱和渺小")
        input_section_parts.append("")
        input_section_parts.append("【转场效果指导】")
        input_section_parts.append("根据剧情节奏、时间跨度、空间变化和情感需要，可以设计以下转场效果：")
        input_section_parts.append("")
        input_section_parts.append("【基础转场】")
        input_section_parts.append("- 硬切（Hard Cut）：直接切换，最常用，干净利落，适用于同一时空的连贯动作")
        input_section_parts.append("- 淡入淡出（Fade In/Fade Out）：画面逐渐变黑/变白再出现，适用于开场、结尾、时间跳跃")
        input_section_parts.append("- 黑屏切换（Black Screen Cut）：瞬间黑屏再亮起，适用于强烈节奏变化、悬念营造")
        input_section_parts.append("- 白屏切换（White Screen Cut）：瞬间白屏再亮起，适用于梦幻、回忆、闪回场景")
        input_section_parts.append("")
        input_section_parts.append("【方向转场】")
        input_section_parts.append("- 左右抽出（Wipe Left/Right）：画面像窗帘一样向左右拉开，适用于空间转换、并行叙事")
        input_section_parts.append("- 上下抽出（Wipe Up/Down）：画面向上下方向拉开，适用于垂直空间转换")
        input_section_parts.append("- 滑动转场（Slide）：画面滑动进入/退出，动感强，适用于快速节奏")
        input_section_parts.append("- 推动转场（Push）：新画面推动旧画面离开，适用于连续运动方向")
        input_section_parts.append("")
        input_section_parts.append("【形状转场】")
        input_section_parts.append("- 圆圈转场（Iris Wipe）：从中心圆点扩大或缩小，适用于聚焦/散焦效果")
        input_section_parts.append("- 方块转场（Box Wipe）：方块形状的划像，适用于科技感、游戏感")
        input_section_parts.append("- 星形转场（Star Wipe）：星形划像，适用于梦幻、魔法效果")
        input_section_parts.append("- 心形转场（Heart Wipe）：心形划像，适用于浪漫场景")
        input_section_parts.append("")
        input_section_parts.append("【创意转场】")
        input_section_parts.append("- 溶解转场（Dissolve）：两个画面重叠融合，适用于时间流逝、回忆、梦幻")
        input_section_parts.append("- 交叉溶解（Cross Dissolve）：前一个画面逐渐消失，后一个逐渐出现，温和过渡")
        input_section_parts.append("- 叠化转场（Morph）：画面元素变形过渡，适用于形状相似的物体转换")
        input_section_parts.append("- 闪光转场（Flash）：快速白光闪烁切换，适用于紧张、爆炸、节奏点")
        input_section_parts.append("- 模糊转场（Blur）：画面模糊后再清晰，适用于梦境、醉酒、眩晕效果")
        input_section_parts.append("- 旋转转场（Spin）：画面旋转切换，适用于时间变化、空间转换")
        input_section_parts.append("- 缩放转场（Zoom）：快速缩放切换，适用于从细节到全景的转换")
        input_section_parts.append("- 碎片转场（Shatter）：画面破碎散落再重组，适用于破坏、爆炸、魔法效果")
        input_section_parts.append("- 翻页转场（Page Turn）：像翻书一样翻页，适用于回忆、故事叙述")
        input_section_parts.append("- 3D转场（3D Transition）：立体空间旋转切换，适用于科技感、现代感")
        input_section_parts.append("")
        input_section_parts.append("【转场选择建议】")
        input_section_parts.append("- 同一时空连贯动作 → 硬切")
        input_section_parts.append("- 时间跨度/季节变化 → 淡入淡出、溶解")
        input_section_parts.append("- 空间位置转换 → 左右抽出、滑动")
        input_section_parts.append("- 紧张/节奏加快 → 闪光、快切")
        input_section_parts.append("- 梦幻/回忆场景 → 溶解、模糊、交叉溶解")
        input_section_parts.append("- 科幻/科技感 → 缩放、3D转场、方块转场")
        input_section_parts.append("- 浪漫/温情 → 溶解、心形转场")
        input_section_parts.append("- 开场/结尾 → 淡入淡出")
        input_section_parts.append("- 悬念/惊吓 → 黑屏切换")
        input_section_parts.append("")
        input_section_parts.append("【节奏控制】")
        input_section_parts.append("- 慢动作（Slow Motion）：强调关键时刻，增加戏剧性")
        input_section_parts.append("- 快切（Quick Cut）：快速切换镜头，增强紧张感和节奏感")
        input_section_parts.append("- 长镜头（Long Take）：连续不剪，展现连贯动作和环境")
        input_section_parts.append("- 定格（Freeze Frame）：画面静止，强调某个瞬间")
        input_section_parts.append("")
        input_section_parts.append("【融合规则 - 重要】")
        input_section_parts.append("1. 【忠于原图】是基础，首帧必须与原图描述一致")
        input_section_parts.append("2. 理解手工提示词的语义意图，据此决定扩展方向、镜头设计和转场效果")
        input_section_parts.append("3. 手工提示词可能包含：")
        input_section_parts.append("   - 动作指令（如：'让角色奔跑'）→ 设计跟拍镜头，展现奔跑过程")
        input_section_parts.append("   - 氛围指令（如：'更压抑的感觉'）→ 设计低角度、阴郁色调、慢镜头")
        input_section_parts.append("   - 节奏指令（如：'快速推进剧情'）→ 设计快节奏剪辑、镜头运动、推拉镜头、闪光转场")
        input_section_parts.append("   - 世界观指令（如：'奇幻冒险风格'）→ 扩展奇幻元素，设计探索性镜头、溶解转场")
        input_section_parts.append("   - 情感指令（如：'孤独感'）→ 设计远景、空旷环境、慢镜头、淡入淡出")
        input_section_parts.append("   - 紧张指令（如：'追逐场景'）→ 设计手持镜头、快速切换、跟拍、黑屏切换")
        input_section_parts.append("   - 时间跨度（如：'从白天到黑夜'）→ 设计淡入淡出、溶解转场")
        input_section_parts.append("   - 空间转换（如：'从室内到室外'）→ 设计左右抽出、滑动转场")
        input_section_parts.append("4. 扩展时要保持逻辑连贯性，不能跳跃太突兀")
        input_section_parts.append("5. 镜头设计和转场效果要服务于剧情和情感表达，不要为了炫技而炫技")
        input_section_parts.append("6. 输出一个完整的视频描述，包含：首帧画面 + 场景扩展 + 动作过程 + 镜头运动 + 转场效果")
        input_section_parts.append("")
        input_section_parts.append("【正确示例 - 忠于原图 + 场景扩展 + 镜头语言 + 转场效果】")
        input_section_parts.append("原图描述: '一个穿红裙的女孩站在沙滩上，阳光明媚，海浪拍打岸边'")
        input_section_parts.append("手工提示词: '让她跑起来，展现海滩的广阔，快速推进剧情'")
        input_section_parts.append("正确输出: '【首帧】特写女孩的背影，她站在沙滩上面朝大海，红裙在海风中轻轻飘动。【镜头运动】镜头开始向后拉远，从中景变为全景。【动作】她开始向前奔跑，镜头切换到跟拍，跟随她的侧脸移动。【场景扩展】随着她的奔跑，镜头慢慢拉远，展现更广阔的海滩：金色的沙滩延伸到远方，左边是礁石群，右边是椰林，海浪一层层拍打。【转场】硬切到航拍视角，女孩越跑越远，变成一个小红点在沙滩上移动。【结尾】极远景俯瞰整个海湾的壮丽景色。'")
        input_section_parts.append("（✅ 首帧忠于原图，扩展了海滩全景，设计了拉远→跟拍→硬切→航拍的镜头序列）")
        input_section_parts.append("")
        input_section_parts.append("【正确示例 - 时间跨越转场】")
        input_section_parts.append("原图描述: '一座古老的城市广场，白天，人们在悠闲地散步'")
        input_section_parts.append("手工提示词: '展现这座城市从白天到夜晚的变化'")
        input_section_parts.append("正确输出: '【首帧】全景展现古老的城市广场，白天阳光明媚，人们在悠闲地散步。【镜头运动】镜头缓慢摇移，扫过广场上的建筑和人群。【转场】淡入淡出过渡，画面渐渐变暗，再渐渐亮起。【场景变化】夜晚的广场，华灯初上，建筑被暖黄色的灯光照亮，人们三三两两坐在长椅上，氛围变得浪漫温馨。'")
        input_section_parts.append("（✅ 使用淡入淡出转场实现时间跨越）")
        input_section_parts.append("")
        input_section_parts.append("【正确示例 - 空间转换转场】")
        input_section_parts.append("原图描述: '一个人在办公室紧张地工作，电脑屏幕亮着'")
        input_section_parts.append("手工提示词: '他决定下班去海边放松'")
        input_section_parts.append("正确输出: '【首帧】中景，一个人在办公室紧张地盯着电脑屏幕，眉头紧锁。【动作】他站起身，拿起外套走向门口。【转场】左右抽出转场，画面像窗帘一样向两边拉开。【场景转换】切换到他站在海边的全景，夕阳西下，海风吹拂，他脸上露出放松的笑容。'")
        input_section_parts.append("（✅ 使用左右抽出转场实现从室内到室外的空间转换）")
        input_section_parts.append("")
        input_section_parts.append("【正确示例 - 紧张追逐转场】")
        input_section_parts.append("原图描述: '一个年轻人在城市街道上行走，周围是高楼大厦'")
        input_section_parts.append("手工提示词: '营造紧张感，他在被追赶'")
        input_section_parts.append("正确输出: '【首帧】年轻人的背面中景，他快步走在街道上。【动作】突然，他听到身后的声音，回头一看，脸上露出惊恐表情（特写）。他开始奔跑，镜头切换到手持跟拍，画面轻微晃动，增强紧迫感。【镜头运动】他穿过人群，镜头快速摇移跟随他的路线。他拐进一条小巷，镜头切换到越肩视角，看到远处两个黑影也在快速接近。【转场】闪光转场，白光一闪。【场景转换】主观视角，画面剧烈晃动，他看到巷子尽头的出口，加快脚步冲出去。'")
        input_section_parts.append("（✅ 使用闪光转场增强紧张感）")
        input_section_parts.append("")
        input_section_parts.append("【正确示例 - 梦幻回忆转场】")
        input_section_parts.append("原图描述: '一个老人坐在公园长椅上，看着手中的老照片'")
        input_section_parts.append("手工提示词: '展现他回忆起年轻时的美好时光'")
        input_section_parts.append("正确输出: '【首帧】特写老人的脸，他低头看着手中的老照片，眼神温柔。【转场】交叉溶解过渡，画面渐渐模糊，照片仿佛活了起来。【场景转换】切换到年轻时的他，在同一片公园里，和爱人一起欢笑奔跑（全景）。阳光透过树叶洒下斑驳光影，画面温暖柔和。【转场】溶解转场回到现实，老人脸上浮现出淡淡的微笑。'")
        input_section_parts.append("（✅ 使用交叉溶解和溶解转场实现回忆效果）")
        input_section_parts.append("")
        input_section_parts.append("【正确示例 - 科幻风格转场】")
        input_section_parts.append("原图描述: '一个实验室里，科学家正在调试一台复杂的机器'")
        input_section_parts.append("手工提示词: '机器启动，展现科幻感'")
        input_section_parts.append("正确输出: '【首帧】近景，科学家专注地调试机器，各种仪表盘闪烁着数据。【动作】他按下启动按钮，机器开始运转，发出嗡嗡声。【转场】缩放转场，镜头快速推近机器核心，画面瞬间变亮。【场景转换】全景，机器完全启动，能量光束在装置中流动，周围的空间似乎开始扭曲，充满未来科技感。'")
        input_section_parts.append("（✅ 使用缩放转场增强科幻感）")
        input_section_parts.append("")
        input_section_parts.append("【错误示例 - 首帧偏离原图，禁止】")
        input_section_parts.append("原图描述: '一个穿红裙的女孩站在沙滩上，阳光明媚'")
        input_section_parts.append("手工提示词: '让她跑起来'")
        input_section_parts.append("错误输出: '一个运动员在田径场上奔跑，镜头特写'")
        input_section_parts.append("（❌ 首帧完全改变了场景和主体，禁止！）")
        input_section_parts.append("")
        input_section_parts.append("【融合示例 - 复杂意图的场景扩展、镜头设计和转场效果】")
        input_section_parts.append("原图描述: '一片宁静的森林，阳光透过树叶洒下斑驳光影，小溪潺潺流过'")
        input_section_parts.append("手工提示词: '这是奇幻冒险的风格，需要主角快速前进，不要停留在当前画面'")
        input_section_parts.append("正确理解: 在保持森林首帧的基础上，扩展奇幻世界观，设计探索性镜头序列和转场效果")
        input_section_parts.append("正确输出: '【首帧】画面从森林地面的特写开始，阳光透过树叶洒下斑驳光影，一只脚踏过溪边的青苔石头。【镜头运动】镜头拉起，展现一个背着行囊的冒险者中景，他环顾四周，眼神坚定。【动作】他沿着小溪快步前行，镜头切换到跟拍，跟随他的脚步移动。【场景扩展】随着他的前进，镜头慢慢拉远，从近景变为全景，展现更广阔的森林：古树参天，藤蔓缠绕，树根盘错，偶尔有萤火虫般的光芒在林间飘动。【转场】硬切，冒险者拨开树枝。【视角切换】镜头切换到越肩视角，前方出现一座古老的石桥，桥下溪水清澈见底，桥的另一端通向迷雾笼罩的深处。【动作】冒险者加快脚步向石桥走去。【镜头运动】镜头慢慢升起，变成俯视视角，展现整片神秘森林的全貌，冒险者变成一个小点走向石桥。【转场】淡入淡出，暗示时间流逝，画面渐渐变暗再亮起。【场景变化】冒险者已经站在石桥上，眺望远方，迷雾中隐约可见城堡的轮廓。'")
        input_section_parts.append("（✅ 首帧忠于原图，扩展了森林的更多细节，设计了特写→拉起→跟拍→拉远→硬切→越肩→俯视→淡入淡出的完整镜头序列和转场效果）")
        input_section_parts.append("")
        input_section_parts.append("【转场效果速查表】")
        input_section_parts.append("| 转场类型 | 适用场景 | 情感效果 |")
        input_section_parts.append("|----------|----------|----------|")
        input_section_parts.append("| 硬切 | 连贯动作、同一时空 | 干净利落、真实 |")
        input_section_parts.append("| 淡入淡出 | 开场/结尾、时间跨越 | 温和、过渡感 |")
        input_section_parts.append("| 黑屏切换 | 悬念、惊吓、节奏变化 | 紧张、戏剧性 |")
        input_section_parts.append("| 左右抽出 | 空间转换、并行叙事 | 动感、流畅 |")
        input_section_parts.append("| 溶解/交叉溶解 | 回忆、梦幻、时间流逝 | 柔和、梦幻 |")
        input_section_parts.append("| 闪光转场 | 紧张、爆炸、节奏点 | 强烈、冲击力 |")
        input_section_parts.append("| 缩放转场 | 细节到全景、科幻 | 科技感、动感 |")
        input_section_parts.append("| 模糊转场 | 梦境、醉酒、眩晕 | 迷幻、不真实 |")
        input_section_parts.append("| 旋转转场 | 时间变化、空间转换 | 动感、眩晕 |")
        input_section_parts.append("| 圆圈转场 | 聚焦/散焦 | 经典、复古 |")
        input_section_parts.append("")
        input_section_parts.append("【镜头语言速查表】")
        input_section_parts.append("| 意图 | 推荐镜头 | 推荐转场 |")
        input_section_parts.append("|------|----------|----------|")
        input_section_parts.append("| 强调情感 | 特写、慢动作 | 溶解 |")
        input_section_parts.append("| 展现环境 | 远景、拉镜头、摇镜头 | 淡入淡出 |")
        input_section_parts.append("| 增强紧张感 | 手持、快切、推镜头 | 闪光、黑屏 |")
        input_section_parts.append("| 展现动作 | 跟拍、中景 | 硬切 |")
        input_section_parts.append("| 代入角色 | 主观视角、越肩 | 无/硬切 |")
        input_section_parts.append("| 展现宏大场面 | 航拍、极远景、升降 | 缩放 |")
        input_section_parts.append("| 时间跨越 | 无特定 | 淡入淡出、溶解 |")
        input_section_parts.append("| 空间转换 | 无特定 | 左右抽出、滑动 |")
        input_section_parts.append("| 回忆/梦幻 | 无特定 | 溶解、模糊 |")
        input_section_parts.append("| 科幻/科技 | 环绕、推拉 | 缩放、方块转场 |")
        input_section_parts.append("| 强调力量 | 低角度 | 无 |")
        input_section_parts.append("| 强调脆弱 | 高角度 | 无 |")
        
    elif has_manual and not has_optional:
        # ========== 只有手工提示词 ==========
        input_section_parts.append("【任务说明 - 从零生成模式】")
        input_section_parts.append("")
        input_section_parts.append("请根据手工提示词，直接生成一个高质量的视频提示词，可以自由设计场景、镜头和转场效果。")
        input_section_parts.append("")
        input_section_parts.append("【手工提示词】：")
        input_section_parts.append(manual_text.strip())
        input_section_parts.append("")
        input_section_parts.append("【镜头和转场设计建议】")
        input_section_parts.append("根据内容需要，可以设计：")
        input_section_parts.append("- 景别变化：特写、近景、中景、全景、远景")
        input_section_parts.append("- 镜头运动：推、拉、摇、移、跟、升降、环绕")
        input_section_parts.append("- 视角选择：主观、客观、越肩、鸟瞰")
        input_section_parts.append("- 节奏控制：慢动作、快切、长镜头")
        input_section_parts.append("- 转场效果：淡入淡出、溶解、左右抽出、闪光、缩放、黑屏切换等")
        
    elif not has_manual and has_optional:
        # ========== 只有原图描述 ==========
        input_section_parts.append("【任务说明 - 优化模式】")
        input_section_parts.append("")
        input_section_parts.append("请基于原图描述，优化成一个更高质量的视频提示词，首帧必须与原图描述一致。")
        input_section_parts.append("可以添加合理的镜头运动、环境扩展和转场效果。")
        input_section_parts.append("")
        input_section_parts.append("【原图描述】：")
        input_section_parts.append(optional_text.strip())
    
    if not input_section_parts:
        return None
    
    input_section = "\n".join(input_section_parts)
    
    if detail_level == "标准":
        strategy = """【详细程度：标准】
- 简洁明了，突出重点动作
- 简单描述镜头变化和转场"""
    elif detail_level == "详细":
        strategy = """【详细程度：详细】
- 包含动作过程、镜头运动、环境氛围、转场效果
- 设计2-3个镜头变化和1-2个转场"""
    else:
        strategy = """【详细程度：极详细】
- 电影级详细描述
- 包含完整的镜头语言设计（景别、运动、视角）
- 包含完整的转场效果设计
- 包含动作细节、光影变化、情绪氛围
- 设计完整的镜头序列和转场序列"""
    
    if output_lang == "中文":
        lang_instruction = "必须使用中文输出"
    else:
        lang_instruction = "必须使用英文输出"
    
    return f"""{header}
{thinking_ban}

你是一个专业的WAN2.2视频提示词专家。

{strategy}

{input_section}

【输出要求】
1. 直接输出视频提示词，不要有任何其他内容
2. 不要输出思考过程、分析、解释
3. 【最高优先级】忠于原图描述，视频首帧必须与原图的核心视觉元素一致
4. 【场景扩展】可以合理扩展原图之外的内容，但要保持风格一致
5. 【镜头语言】根据剧情需要设计合适的镜头运动、景别变化和视角切换
6. 【转场效果】根据剧情节奏、时间跨度、空间转换和情感需要，设计合适的转场效果
7. {lang_instruction}

直接输出视频提示词："""

def _format_prompt_interrogation(image_base64: str, detail_level: str, output_lang: str) -> str:
    """格式化提示词反推提示词"""
    header = "【绝对指令】直接输出最终结果，严禁任何思考过程、分析或解释。"
    
    thinking_ban = """
【严重警告】
- 绝对不允许输出任何思考过程、分析或解释
- 直接输出反推的提示词，不要有任何前缀、后缀或标签
- 只输出提示词本身
"""
    
    if detail_level == "标准":
        strategy = """简洁描述主要元素和场景"""
    elif detail_level == "详细":
        strategy = """详细描述元素、光线、色彩、风格"""
    else:
        strategy = """极致详细地描述所有视觉元素，包括光影、质感、构图、氛围"""
    
    if output_lang == "中文":
        lang_instruction = "必须使用中文输出"
    else:
        lang_instruction = "必须使用英文输出"
    
    return f"""{header}
{thinking_ban}

你是一个专业的AI绘画提示词反推专家。请仔细观察图片，识别图片中的内容，并生成高质量的提示词。

【反推要求】
{strategy}

【输出要求】
1. 直接输出反推的提示词，不要有任何其他内容
2. 不要输出思考过程、分析、解释
3. 不要添加图片中没有的内容
4. {lang_instruction}

直接输出提示词："""

# ========== 节点类 ==========

class AIConnector:
    @classmethod
    def INPUT_TYPES(cls):
        cfg = _CONFIG
        return {
            "required": {
                "地址": ("STRING", {"default": cfg["base_url"], "placeholder": "http://127.0.0.1:1234"}),
                "令牌": ("STRING", {"default": cfg["token"], "placeholder": "API令牌（可选）"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("AI配置",)
    FUNCTION = "connect"
    CATEGORY = "AI"

    def connect(self, **kwargs):
        addr = kwargs["地址"]
        token = kwargs["令牌"]
        _save_config(base_url=addr, token=token)
        config_json = json.dumps({"address": addr, "token": token})
        return (config_json,)

class AIPromptInterrogator:
    @classmethod
    def INPUT_TYPES(cls):
        cfg = _CONFIG
        last = cfg.get("last_model", "") or ""
        return {
            "required": {
                "图片": ("IMAGE",),
                "AI配置": ("STRING", {"forceInput": True, "placeholder": "连接AI连接器的输出"}),
                "模型": ("STRING", {"default": last, "placeholder": "输入多模态模型名称"}),
                "详细程度": (["标准", "详细", "极详细"], {"default": "详细"}),
                "输出语言": (["英文", "中文"], {"default": "英文"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("反推提示词",)
    FUNCTION = "interrogate"
    CATEGORY = "AI"

    def interrogate(self, **kwargs):
        image_tensor = kwargs["图片"]
        config_json = kwargs["AI配置"]
        model = kwargs["模型"].strip()
        detail_level = kwargs["详细程度"]
        output_lang = kwargs["输出语言"]
        
        try:
            config = json.loads(config_json)
            addr = config.get("address", "")
            token = config.get("token", "")
        except:
            return ("AI配置格式错误，请重新连接AI连接器",)
        
        if not addr or not token:
            return ("请先连接AI服务",)
        if not model:
            return ("请输入多模态模型名称",)
        
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
            
            req = _format_prompt_interrogation(img_base64, detail_level, output_lang)
            
            res = _ai_chat(addr, token, model, req, _CONFIG["timeout"], img_base64)
            res = res.strip().strip('"\'')
            
            # 清理可能残留的标签 - 只删除开头的标签，保留完整内容
            res = re.sub(r'^(反推的提示词:|思考[：:]|分析[：:]|让我想想|首先|然后|最后|综上所述|好的|我理解了|明白了)\s*', '', res, flags=re.I)
            
            # 反推提示词通常是单行，但如果有多行也保留完整
            if not res or len(res) < 5:
                lines = [line.strip() for line in res.split('\n') if line.strip()]
                if lines:
                    res = lines[0]
            
            return (res,)
        except Exception as e:
            return (f"提示词反推错误: {str(e)}",)

class AIImagePromptConverter:
    @classmethod
    def INPUT_TYPES(cls):
        cfg = _CONFIG
        last = cfg.get("last_model", "") or ""
        return {
            "required": {
                "AI配置": ("STRING", {"forceInput": True, "placeholder": "连接AI连接器的输出"}),
                "手工提示词": ("STRING", {"multiline": True, "placeholder": "输入你的需求/意图/修改指令..."}),
                "图片反推描述": ("STRING", {"multiline": True, "placeholder": "输入图片反推描述（原图描述）"}),
                "模型": ("STRING", {"default": last, "placeholder": "输入模型名称"}),
                "生成模式": (["文生图", "图生图"], {"default": "文生图"}),
                "详细程度": (["标准", "详细", "极详细"], {"default": "详细"}),
                "提示类型": (["正向", "负向"], {"default": "正向"}),
                "输出语言": (["英文", "中文"], {"default": "英文"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("图片提示词",)
    FUNCTION = "convert_image"
    CATEGORY = "AI"

    def convert_image(self, **kwargs):
        config_json = kwargs["AI配置"]
        manual_prompt = kwargs["手工提示词"].strip()
        optional_prompt = kwargs["图片反推描述"].strip()
        model = kwargs["模型"].strip()
        mode = kwargs["生成模式"]
        detail_level = kwargs["详细程度"]
        prompt_type = kwargs["提示类型"]
        output_lang = kwargs["输出语言"]
        
        try:
            config = json.loads(config_json)
            addr = config.get("address", "")
            token = config.get("token", "")
        except:
            return ("AI配置格式错误，请重新连接AI连接器",)
        
        if not manual_prompt and not optional_prompt:
            return ("请填写手工提示词或图片反推描述",)
        if not addr or not token:
            return ("请先连接AI服务",)
        if not model:
            return ("请输入模型名称",)
        
        req = _format_image_prompt(manual_prompt, optional_prompt, mode, detail_level, prompt_type, output_lang)
        key = (manual_prompt, optional_prompt, model, mode, detail_level, prompt_type, output_lang)
        
        if key in _CACHE:
            return (_CACHE[key],)
        
        try:
            res = _ai_chat(addr, token, model, req, _CONFIG["timeout"])
            res = res.strip().strip('"\'')
            
            # 清理可能残留的标签 - 只删除开头的标签，保留完整内容
            res = re.sub(r'^(转换后的提示词:|Converted prompt:|思考[：:]|分析[：:]|让我想想|首先|然后|最后|综上所述|好的|我理解了|明白了)\s*', '', res, flags=re.I)
            
            # 图片提示词通常是单行，但如果有多行也保留完整
            if not res or len(res) < 5:
                lines = [line.strip() for line in res.split('\n') if line.strip()]
                if lines:
                    res = lines[0]
            
            _CACHE[key] = res
            return (res,)
        except Exception as e:
            return (str(e),)

class AIVideoPromptConverter:
    @classmethod
    def INPUT_TYPES(cls):
        cfg = _CONFIG
        last = cfg.get("last_model", "") or ""
        return {
            "required": {
                "AI配置": ("STRING", {"forceInput": True, "placeholder": "连接AI连接器的输出"}),
                "手工提示词": ("STRING", {"multiline": True, "placeholder": "输入你的需求/意图/修改指令..."}),
                "图片反推描述": ("STRING", {"multiline": True, "placeholder": "输入图片反推描述（原图描述）"}),
                "模型": ("STRING", {"default": last, "placeholder": "输入模型名称"}),
                "生成模式": (["文生视频", "图生视频"], {"default": "文生视频"}),
                "详细程度": (["标准", "详细", "极详细"], {"default": "详细"}),
                "输出语言": (["中文", "英文"], {"default": "中文"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("视频提示词",)
    FUNCTION = "convert_video"
    CATEGORY = "AI"

    def convert_video(self, **kwargs):
        config_json = kwargs["AI配置"]
        manual_prompt = kwargs["手工提示词"].strip()
        optional_prompt = kwargs["图片反推描述"].strip()
        model = kwargs["模型"].strip()
        mode = kwargs["生成模式"]
        detail_level = kwargs["详细程度"]
        output_lang = kwargs["输出语言"]
        
        try:
            config = json.loads(config_json)
            addr = config.get("address", "")
            token = config.get("token", "")
        except:
            return ("AI配置格式错误，请重新连接AI连接器",)
        
        if not manual_prompt and not optional_prompt:
            return ("请填写手工提示词或图片反推描述",)
        if not addr or not token:
            return ("请先连接AI服务",)
        if not model:
            return ("请输入模型名称",)
        
        video_req = _format_video_prompt(manual_prompt, optional_prompt, mode, detail_level, output_lang)
        
        key = (manual_prompt, optional_prompt, model, mode, detail_level, output_lang)
        
        if key in _CACHE:
            return (_CACHE[key],)
        
        try:
            res = _ai_chat(addr, token, model, video_req, _CONFIG["timeout"])
            res = res.strip().strip('"\'')
            
            # 清理可能残留的标签 - 只删除开头的标签，保留完整内容
            res = re.sub(r'^(处理后的[^:]*:|优化后[^:]*:|视频提示词[^:]*:|思考[：:]|分析[：:]|让我想想|首先|然后|最后|综上所述|好的|我理解了|明白了)\s*', '', res, flags=re.I)
            
            # 重要：视频提示词通常是多行内容（包含分镜描述），不要截断！
            # 只有在内容为空时才尝试取第一行作为后备
            if not res or len(res) < 10:
                lines = [line.strip() for line in res.split('\n') if line.strip() and not line.startswith(('```', '`'))]
                if lines:
                    res = lines[0]
            
            _CACHE[key] = res
            return (res,)
        except Exception as e:
            return (f"错误: {str(e)}",)

NODE_CLASS_MAPPINGS = {
    "AIConnector": AIConnector,
    "AIPromptInterrogator": AIPromptInterrogator,
    "AIImagePromptConverter": AIImagePromptConverter,
    "AIVideoPromptConverter": AIVideoPromptConverter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIConnector": "🤖 AI 连接器",
    "AIPromptInterrogator": "🔍 AI 提示词反推",
    "AIImagePromptConverter": "🎨 AI 图片提示词",
    "AIVideoPromptConverter": "🎬 AI 视频提示词",
}