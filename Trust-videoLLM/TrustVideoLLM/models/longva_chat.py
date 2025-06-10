import os
from typing import List
from PIL import Image
import torch
from TrustVideoLLM.models.base import BaseChat, Response
from TrustVideoLLM.utils.registry import registry
from PIL import Image
from TrustVideoLLM.models.longva.model.builder import load_pretrained_model
from TrustVideoLLM.models.longva.mm_utils import tokenizer_image_token, process_images
from TrustVideoLLM.models.longva.constants import IMAGE_TOKEN_INDEX
from decord import VideoReader, cpu
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import numpy as np



@registry.register_videoLLM()
class LongVA(BaseChat):

    model_family = {
        'LongVA-7B':'lmms-lab/LongVA-7B',
        'LongVA-7B-TPO':'ruili0/LongVA-7B-TPO'
    }
   

    def __init__(self, model_id: str, device: str = "cuda:0"):
        super().__init__(model_id)
        self.device = device
        model_id = self.model_family[self.model_id]
       
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(model_id, None, "llava_qwen", device_map="cuda:0")

    

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

        gen_kwargs = {"do_sample": False, "temperature": 0.5, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": max_new_tokens}


        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

        print('video_path=', video_path)
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()


        video_tensor = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(self.device, dtype=torch.float16)
        with torch.inference_mode():
            output_ids = self.model.generate(input_ids, images=[video_tensor],  modalities=["video"], **gen_kwargs)
        response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()



        scores = None

        return Response(self.model_id, response, scores, None, extra)
       


