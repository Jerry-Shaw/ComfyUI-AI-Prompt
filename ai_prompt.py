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

def _ai_chat(url: str, token: str, model: str, msg: str, timeout: int = 60, image_base64: str = None) -> str:
    """通用的AI聊天接口，支持OpenAI兼容的API，支持多模态"""
    base = url.rstrip('/')
    
    if image_base64:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": msg},
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
                    return choice["message"]["content"].strip()
                elif "text" in choice:
                    return choice["text"].strip()
                elif "content" in choice:
                    return choice["content"].strip()
            
            if "output" in data and len(data["output"]) > 0:
                if "content" in data["output"][0]:
                    return data["output"][0]["content"].strip()
            
            if "response" in data:
                return data["response"].strip()
            
            return str(data)
        except Exception as e:
            pass
    
    endpoints = [
        {"url": f"{base}/api/v1/chat", "payload": {"model": model, "input": msg, "temperature": 0.1}},
        {"url": f"{base}/v1/chat/completions", "payload": {"model": model, "messages": [{"role": "user", "content": msg}], "temperature": 0.1}}
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
            elif "output" in data and len(data["output"]) > 0:
                if "content" in data["output"][0]:
                    res = data["output"][0]["content"].strip()
                else:
                    res = str(data["output"][0]).strip()
            else:
                res = str(data).strip()
            
            _save_config(model=model)
            return res
        except:
            continue
    
    raise Exception("连接失败")

def _format_image_prompt(manual_text: str, optional_text: str, mode: str, detail_level: str, prompt_type: str, output_lang: str) -> str:
    """格式化图片提示词"""
    header = "【绝对指令】直接输出最终结果，严禁任何思考过程、分析或解释。"
    
    if optional_text:
        input_section = f"""【图片识别内容】（来自提示词反推节点）
{optional_text}

【用户手动输入】（用户补充或修改的内容）
{manual_text}

【处理要求】
请将以上两部分内容合并优化，生成完整的提示词。优先保留图片识别的核心元素，同时融入用户手动输入的要求。"""
    else:
        input_section = f"""【用户输入】
{manual_text}

【处理要求】
请根据用户输入生成提示词。"""
    
    if detail_level == "标准":
        strategy = """【处理策略：严格遵循原文】
- 只做必要的语法修正和错别字修正
- 补全描述中明显缺失的关键元素（如主体缺失、风格缺失）
- 严禁自行添加原文没有的场景、环境、细节
- 保持原提示词的核心内容和风格不变
- 输出长度与原文相近"""
    elif detail_level == "详细":
        strategy = """【处理策略：适度补充】
- 在原文基础上进行合理的细节补充
- 可补充光影、质感、构图等与原文相关的内容
- 不添加原文完全未提及的新元素
- 描述可适当细化，但不改变原意
- 输出长度约为原文的1.5-2倍"""
    else:
        strategy = """【处理策略：全面丰富】
- 可充分发挥想象力，对原文进行全方位的丰富和扩展
- 添加精致的视觉细节：光影、质感、色彩、构图、氛围
- 补充完整的场景环境和情绪基调
- 追求电影级的画面效果
- 输出长度不限，追求极致细节"""
    
    if mode == "文生图":
        if prompt_type == "正向":
            mode_guide = """【文生图正向提示词模式】
从零构建画面，侧重：
- 完整的场景设定和元素构建
- 主体形象的详细刻画
- 风格、构图、光影的明确描述"""
        else:
            mode_guide = """【文生图负向提示词模式】
排除不需要的元素，侧重：
- 列出需要避免的瑕疵和缺陷
- 排除不符合预期的元素
- 只针对用户输入中明确提到的负面内容
- 不添加用户未提及的负面元素"""
    else:
        if prompt_type == "正向":
            mode_guide = """【图生图正向提示词模式】
基于参考图进行微调和优化，侧重：
- 保留原图中的核心元素和风格
- 在原有基础上进行细节增强和优化
- 轻微调整构图、光影、色彩
- 保持与原图的连贯性和一致性"""
        else:
            mode_guide = """【图生图负向提示词模式】
基于原图排除不需要的元素，侧重：
- 避免出现原提示词中未提及的新元素
- 排除会破坏原图结构的负面因素
- 保持与原图的一致性
- 只排除明确需要避免的问题"""
    
    if output_lang == "中文":
        lang_instruction = "必须使用中文输出"
        output_format = "输出中文提示词"
    else:
        lang_instruction = "必须使用英文输出"
        output_format = "输出英文提示词"
    
    return f"""{header}

你是一个专业的Stable Diffusion提示词转换专家。
请将以下用户输入转换为高质量的{prompt_type}提示词，用于AI绘画。

【核心原则】
严格围绕用户输入的原始内容进行转换，不要自行添加新元素

【生成模式】
{mode_guide}

{strategy}

【最重要规则 - 必须绝对遵守】
1. 严禁任何形式的思考过程、分析、解释或额外说明
2. 禁止使用"让我想想"、"首先"、"然后"、"综上所述"、"接下来"、"我认为"等任何思考性词语
3. 禁止输出任何标签，如"思考过程："、"分析："、"转换后的提示词："、"结果："等
4. 禁止输出任何解释性文字
5. 直接输出最终的提示词，不要有任何前缀或后缀
6. 不要添加用户输入中没有描述的元素
7. {lang_instruction}

【输出格式】
直接输出{output_format}，只输出提示词本身，不包含任何其他内容。

{input_section}

直接输出提示词："""

def _format_video_prompt(manual_text: str, optional_text: str, mode: str, detail_level: str, output_lang: str) -> str:
    """格式化视频提示词"""
    header = "【绝对指令】直接输出最终结果，严禁任何思考过程、分析或解释。"
    
    if optional_text:
        input_section = f"""【图片识别内容】（来自提示词反推节点）
{optional_text}

【用户手动输入】（用户补充或修改的内容）
{manual_text}

【处理要求】
请将以上两部分内容合并优化，生成完整的视频提示词。优先保留图片识别的核心元素和动作，同时融入用户手动输入的要求。"""
    else:
        input_section = f"""【用户输入】
{manual_text}

【处理要求】
请根据用户输入生成视频提示词。"""
    
    if detail_level == "标准":
        strategy = """【处理策略：严格遵循原文】
- 只做必要的语法修正和错别字修正
- 补全描述中明显缺失的关键元素（如主体缺失、动作缺失）
- 严禁自行添加原文没有的场景、环境、细节
- 保持原提示词的核心内容和风格不变
- 输出长度与原文相近"""
    elif detail_level == "详细":
        strategy = """【处理策略：适度补充】
- 在原文基础上进行合理的细节补充
- 可补充环境氛围、光线、情绪等与原文相关的内容
- 不添加原文完全未提及的新场景元素
- 动作描述可适当细化，但不改变原意
- 输出长度约为原文的1.5-2倍"""
    else:
        strategy = """【处理策略：全面丰富（电影级）】
- 可充分发挥想象力，对原文进行全方位的丰富和扩展
- 添加电影级的视觉细节：光影、质感、色彩、构图
- 补充完整的场景环境和氛围营造
- 细化动作过程的起承转合
- 可添加镜头语言描述（推拉摇移、景别变化）
- 营造沉浸式的情绪氛围
- 输出长度不限，追求极致细节"""
    
    if mode == "文生视频":
        mode_guide = """【文生视频模式说明】
从零构建视觉场景，侧重：
- 完整的场景设定和环境构建
- 主体形象的详细刻画
- 动作的完整过程描述
- 时间流逝和变化过程"""
    else:
        mode_guide = """【图生视频模式说明】
基于参考图生成视频，侧重：
- 动作的流畅性和连续性
- 主体在画面中的动态变化
- 场景的微妙变化而非剧烈转变
- 保持与参考图的一致性"""
    
    if output_lang == "中文":
        lang_instruction = "必须使用中文输出"
    else:
        lang_instruction = "必须使用英文输出"
    
    return f"""{header}

你是一个专业的WAN2.2视频生成提示词优化专家。你的任务是根据用户输入的简要描述，生成高质量的视频提示词。

【核心任务】
{mode_guide}

{strategy}

【最重要规则 - 必须绝对遵守】
1. 严禁任何形式的思考过程、分析、解释或额外说明
2. 禁止使用"让我想想"、"首先"、"然后"、"综上所述"、"接下来"、"我认为"等任何思考性词语
3. 禁止输出任何标签，如"优化后："、"润色后："、"视频提示词："、"结果："等
4. 禁止输出任何解释性文字
5. 直接输出最终的视频提示词，不要有任何前缀或后缀
6. 输出内容为纯文本，不要使用markdown格式
7. {lang_instruction}

【输出格式】
直接输出视频提示词本身，只输出提示词，不包含任何其他内容。

{input_section}

直接输出视频提示词："""

def _format_prompt_interrogation(image_base64: str, detail_level: str, output_lang: str) -> str:
    """格式化提示词反推提示词"""
    header = "【绝对指令】直接输出最终结果，严禁任何思考过程、分析或解释。"
    
    if detail_level == "标准":
        strategy = """【反推策略：简洁描述】
- 识别图片中的主要元素和主体
- 简要描述场景、风格、光线
- 输出简洁的关键词形式"""
    elif detail_level == "详细":
        strategy = """【反推策略：详细描述】
- 详细识别图片中的各个元素
- 描述场景氛围、光线效果、色彩搭配
- 识别艺术风格和构图特点"""
    else:
        strategy = """【反推策略：极致详细】
- 全方位识别图片中的所有细节
- 分析光影、质感、色彩层次
- 识别艺术风格、技术手法
- 描述情绪氛围和视觉焦点"""
    
    if output_lang == "中文":
        lang_instruction = "必须使用中文输出"
        output_format = "输出中文提示词"
    else:
        lang_instruction = "必须使用英文输出"
        output_format = "输出英文提示词"
    
    return f"""{header}

你是一个专业的AI绘画提示词反推专家。请仔细观察图片，识别图片中的内容，并生成高质量的提示词。

【反推任务】
{strategy}

【最重要规则 - 必须绝对遵守】
1. 严禁任何形式的思考过程、分析、解释或额外说明
2. 禁止使用"让我想想"、"首先"、"然后"、"综上所述"、"接下来"、"我认为"等任何思考性词语
3. 禁止输出任何标签，如"反推结果："、"识别结果："、"提示词："、"结果："等
4. 禁止输出任何解释性文字
5. 直接输出反推的提示词，不要有任何前缀或后缀
6. 不要添加图片中没有的内容
7. {lang_instruction}

【输出格式】
直接输出{output_format}，只输出提示词本身，不包含任何其他内容。

请直接输出反推的提示词："""

class AIConnector:
    """AI连接器 - 统一管理地址和令牌"""
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
    """AI提示词反推节点 - 使用多模态LLM识别图片内容并生成提示词"""
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
            
            res = re.sub(r'^(反推的提示词:|思考[：:]|分析[：:]|让我想想|首先|然后|最后|综上所述)[\s:]*', '', res, flags=re.I)
            
            if '\n' in res:
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
                "图片提示词": ("STRING", {"multiline": True, "placeholder": "输入图片提示词..."}),
                "图片反推描述（可选）": ("STRING", {"multiline": True, "placeholder": "可选，输入图片反推描述或手动填写"}),
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
        manual_prompt = kwargs["图片提示词"].strip()
        optional_prompt = kwargs.get("图片反推描述（可选）", "").strip()
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
            return ("请输入图片提示词或填写图片反推描述（可选）",)
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
            
            res = re.sub(r'^(转换后的提示词:|Converted prompt:|思考[：:]|分析[：:]|让我想想|首先|然后|最后|综上所述)[\s:]*', '', res, flags=re.I)
            
            if '\n' in res:
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
                "视频提示词": ("STRING", {"multiline": True, "placeholder": "输入视频提示词..."}),
                "图片反推描述（可选）": ("STRING", {"multiline": True, "placeholder": "可选，输入图片反推描述或手动填写"}),
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
        manual_prompt = kwargs["视频提示词"].strip()
        optional_prompt = kwargs.get("图片反推描述（可选）", "").strip()
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
            return ("请输入视频提示词或填写图片反推描述（可选）",)
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
            
            res = re.sub(r'^(处理后的[^:]*:|优化后[^:]*:|视频提示词[^:]*:|思考[：:]|分析[：:]|让我想想|首先|然后|最后|综上所述)[\s:]*', '', res, flags=re.I)
            
            if '\n' in res:
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