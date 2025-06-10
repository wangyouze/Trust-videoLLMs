



from http import HTTPStatus
import os
import dashscope
from typing import List
import torch
from TrustVideoLLM.models.base import BaseChat, Response
from TrustVideoLLM.utils.registry import registry
from openai import OpenAI
from decord import VideoReader, cpu 
from PIL import Image
from io import BytesIO
import base64


def pil_to_base64(img):
    # 创建一个字节流缓冲区
    buffered = BytesIO()
    # 将图像保存到缓冲区（格式可以是 'PNG', 'JPEG' 等）
    img.save(buffered, format="PNG")
    # 获取字节数据并编码为 base64
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_base64
    
def encode_video(video_path, max_frames_num):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > max_frames_num:
        frame_idx = uniform_sample(frame_idx, max_frames_num)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    frames = [pil_to_base64(v) for v in frames]
    print('num frames:', len(frames))
    return frames


@registry.register_videoLLM()
class Qwen_VL_MAX(BaseChat):
    model_family = {
        'Qwen-VL-MAX':'Qwen-VL-MAX'
    }

    def __init__(self, model_id: str, device: str = "cuda:0"):
        super().__init__(model_id)
        self.device = device
        self.model_id = model_id

        self.client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key="",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
       
    

    @torch.no_grad()
    def chat(self, messages: List, **generation_kwargs):
        # TODO: if system message provided.
        assert len(messages) == 1, "Only support one-turn conversation currently"
        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:
                if message["role"] == "user":
                    if isinstance(message["content"], dict):
                        # multimodal
                        user_message = message["content"]["question"]
                        video_path = message['content']['video_path']
                        
                    else:
                        user_message = message["content"]
                        
                elif message["role"] == "assistant":
                    # TODO: add assistant answer into the conversation
                    pass
            else:
                raise ValueError(
                    "Unsupported role. Only system, user and assistant are supported."
                )

        max_new_tokens = generation_kwargs['max_new_tokens']

        video = encode_video(video_path, max_frames_num=128)

        while True:
            completion = self.client.chat.completions.create(
            model="qwen-vl-max-latest",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video
                    },
                    {
                        "type": "json_object",
                        "text": user_message
                    }]}]
                )
        

            print(completion.model_dump_json()["choices"]['message'])


        scores = None

        return Response(self.model_id, response, scores, None)
       


