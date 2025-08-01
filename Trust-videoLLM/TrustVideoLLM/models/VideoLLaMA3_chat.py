import os
from typing import List
from PIL import Image
import torch
from TrustVideoLLM.models.base import BaseChat, Response
from TrustVideoLLM.utils.registry import registry
import cv2
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel, AutoImageProcessor
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

class VideoReaderFromTensor:
    def __init__(self, video_tensor):
        # video_tensor 的形状假设为 (num_frames, height, width, channels)
        self.video_tensor = video_tensor
        self.total_frames = video_tensor.shape[0]
        self.avg_fps = 30  # 假设视频的平均帧率为 30 FPS，你可以根据实际情况调整
    
    def get_batch(self, frame_indices):
        # 从 video_tensor 中按给定的 frame_indices 提取对应的帧
        return self.video_tensor[frame_indices]
    
    def get_avg_fps(self):
        return self.avg_fps
    
    def __len__(self):
        return self.total_frames


def encode_video(video_tensor, max_num_frames):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReaderFromTensor(video_tensor)
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > max_num_frames:
        frame_idx = uniform_sample(frame_idx, max_num_frames)
    frames = vr.get_batch(frame_idx)
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames

def convert_to_video(video_frames, fps, output_path):
    height, width, _ = video_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 'mp4v' 编码

    # 创建 VideoWriter 对象
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in video_frames:
        # 将帧从 RGB 转换回 BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    # 释放 VideoWriter 对象
    out.release()

def save_video(video_array, output_path, fps=30):
        
    T, H, W, C = video_array.shape
    # OpenCV 使用 BGR 格式，如果输入是 RGB，则需要转换
    if C == 3:
        video_array = video_array[:, :, :, ::-1]  # 将 RGB 转换为 BGR
    # 定义视频编码器（例如 MP4 格式）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    # 写入每一帧
    for frame in video_array:
        out.write(frame)
    out.release()


@registry.register_videoLLM()
class VideoLLaMA3(BaseChat):

    model_family = {
        'VideoLLaMA3-7B':'DAMO-NLP-SG/VideoLLaMA3-7B'
    }
    def __init__(self, model_id: str, device: str = "cuda:0"):
        super().__init__(model_id)
        self.device = device
        model_id = self.model_family[self.model_id]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    

    @torch.no_grad()
    def chat(self, messages: List, **generation_kwargs):
        # TODO: if system message provided.
        assert len(messages) == 1, "Only support one-turn conversation currently"
        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:
                if message["role"] == "user":
                    if isinstance(message["content"], dict):
                        # multimodal
                        video = message["content"]["video_frames"]
                        user_message = message["content"]["question"]
                        video_path = message['content']['video_path']
                        extra = message["content"]["extra"]
                        
                    else:
                        user_message = message["content"]
                        
                elif message["role"] == "assistant":
                    # TODO: add assistant answer into the conversation
                    pass
            else:
                raise ValueError(
                    "Unsupported role. Only system, user and assistant are supported."
                )
            
        max_frames_num = 64
        max_new_tokens = generation_kwargs['max_new_tokens']
        
        # Video conversation
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": max_frames_num}},
                    {"type": "text", "text": user_message},
                ]
            },
        ]

        inputs = self.processor(conversation=conversation, return_tensors="pt").to(self.model.device)
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.device, dtype=torch.bfloat16)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


        scores = None

        return Response(self.model_id, response, scores, None, extra)
       


