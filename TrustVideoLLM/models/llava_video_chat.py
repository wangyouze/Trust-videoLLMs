
from typing import List
import torch
from TrustVideoLLM.models.base import BaseChat, Response
from TrustVideoLLM.utils.registry import registry
from TrustVideoLLM.models.llava.model.builder import load_pretrained_model
from TrustVideoLLM.models.llava.mm_utils import tokenizer_image_token
from TrustVideoLLM.models.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from TrustVideoLLM.models.llava.conversation import conv_templates
from decord import VideoReader, cpu 
import numpy as np
import copy

class VideoReaderFromTensor:
    def __init__(self, video):
        self.video = video
        self.total_frames = video.shape[0]
        self.avg_fps = 30  # 假设视频的平均帧率为 30 FPS，你可以根据实际情况调整
    
    def get_batch(self, frame_indices):
        # 从 video_tensor 中按给定的 frame_indices 提取对应的帧
        return self.video[frame_indices]
    
    def get_avg_fps(self):
        return self.avg_fps
    
    def __len__(self):
        return self.total_frames


def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    """
    加载视频并采样固定帧数
    Args:
        video_path (str): 视频文件路径
        max_frames_num (int): 最大采样帧数
        fps (int): 每秒采样帧数
        force_sample (bool): 是否强制均匀采样

    Returns:
        tuple: 视频帧数组、帧时间信息、视频时长
    """

    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
   
    return spare_frames, frame_time, video_time





@registry.register_videoLLM()
class LLaVA_VIDEO(BaseChat):

    # TODO: update model config
    MODEL_CONFIG = {
        "LLaVA-Video-7B-Qwen2":"lmms-lab/LLaVA-Video-7B-Qwen2",
        "LLaVA-Video-72B-Qwen2":"lmms-lab/LLaVA-Video-72B-Qwen2",
    }


    def __init__(self, model_id: str, device: str = "cuda"):
        super().__init__(model_id)
        self.device = "cuda"
        model_id = self.model_family[self.model_id]

        device_map = "auto"
        model_name = "llava_qwen"

        self.tokenizer, self.model, self.processor, self.max_length = load_pretrained_model(
            model_id, None, model_name, torch_dtype="bfloat16", device_map=device_map, 
        )
        self.model.eval()


    @torch.no_grad()
    def chat(self, messages: List, **generation_kwargs):
        # TODO: if system message provided.
        assert len(messages) == 1, "Only support one-turn conversation currently"
        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:
                if message["role"] == "user":
                    if isinstance(message["content"], dict):
                        # multimodal
                        video_path = message["content"]["video_path"]
                        user_message = message["content"]["question"]
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


        max_new_tokens = generation_kwargs['max_new_tokens']
        max_frames_num = 64

        video, frame_time, video_time = load_video(video_path, max_frames_num=max_frames_num)
        video_tensor = torch.from_numpy(video).to(self.device)
        video_tensor = self.processor.preprocess(video, return_tensors="pt")["pixel_values"].to(self.device).bfloat16()
        video_tensor = [video_tensor]


        question = DEFAULT_IMAGE_TOKEN + generation_kwargs['system'] + user_message
        conv_template = "qwen_1_5"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

       
        with torch.no_grad():
            cont = self.model.generate(
                input_ids,
                images=video_tensor,
                modalities=["video"],
                do_sample=False,
                temperature=0,
                max_new_tokens=max_new_tokens,
            )
        output = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

        scores = None

        del video, video_tensor, input_ids
        return Response(self.model_id, output, scores, None, extra)