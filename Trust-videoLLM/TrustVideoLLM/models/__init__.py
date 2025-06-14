from TrustVideoLLM.utils.registry import registry
from TrustVideoLLM.models.base import BaseChat, Response
from typing import List
from TrustVideoLLM.models.llava_video_chat import LLaVA_VIDEO
from TrustVideoLLM.models.mPlug_Owl3_chat import mPlug_Owl3
from TrustVideoLLM.models.long_llava_chat import Long_LLaVA
from TrustVideoLLM.models.MiniCPM import MiniCPM
from TrustVideoLLM.models.VideoLLaMA3_chat import VideoLLaMA3
from TrustVideoLLM.models.longva_chat import LongVA
from TrustVideoLLM.models.openai_chat import OpenAIChat
from TrustVideoLLM.models.DeepSeek_chat import DeepSeekChat
# from TrustVideoLLM.models.Qwen_VL_max import Qwen_VL_MAX
from TrustVideoLLM.models.ShareGPT4Video_chat import ShareGPT4Video
from TrustVideoLLM.models.video_chatgpt_chat import Video_ChatGPT
from TrustVideoLLM.models.Qwen2_5_VL_chat import Qwen2_5_VL 
from TrustVideoLLM.models.Oryx_chat import Oryx
from TrustVideoLLM.models.GPT4O_chat import OpenAIChat
from TrustVideoLLM.models.InternVL2_5_chat import InternVL2_5
from TrustVideoLLM.models.llava_onevision_chat import Llava_OneVision

def load_chatmodel(model_id: str, device: str = "cuda:0") -> "BaseChat":
    # print_registered_models()
    return registry.get_chatmodel_class(model_id)(model_id=model_id, device=device)


def model_zoo() -> List["BaseChat"]:
    return registry.list_chatmodels()

# def print_registered_models():
#     # 假设 registry 是一个全局对象或已导入的对象
#     if hasattr(registry, '_registry'):
#         # 如果 registry 使用 _registry 属性存储模型
#         registered_models = registry._registry.keys()
#     elif hasattr(registry, 'registry'):
#         # 如果 registry 使用 registry 属性存储模型
#         registered_models = registry.registry.keys()
#     else:
#         raise AttributeError("无法确定 registry 的结构，请检查其实现")

#     print("已注册的模型:")
#     for model_id in registered_models:
#         print(f"- {model_id}")
