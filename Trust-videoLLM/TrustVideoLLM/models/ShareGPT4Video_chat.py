import os
from typing import List
from PIL import Image
import torch
from TrustVideoLLM.models.base import BaseChat, Response
from TrustVideoLLM.utils.registry import registry
from PIL import Image
from TrustVideoLLM.models.share4video.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from TrustVideoLLM.models.share4video.conversation import conv_templates
from TrustVideoLLM.models.share4video.mm_utils import (get_model_name_from_path, process_images,
                            tokenizer_image_token)

from decord import VideoReader, cpu 


def encode_video(video_path, max_frames_num):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > max_frames_num:
        frame_idx = uniform_sample(frame_idx, max_frames_num)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames


@registry.register_videoLLM()
class ShareGPT4Video(BaseChat):

    model_family = {
        'sharegpt4video-8b': "Lin-Chen/sharegpt4video-8b"
    }
    def __init__(self, model_id: str, device: str = "cuda:0"):
        super().__init__(model_id)
        self.device = device
        model_id = self.model_family[self.model_id]
        model_id = "/data1/home/wangyouze/projects/checkpoints/sharegpt4video-8b/"

        model_name = get_model_name_from_path(model_id)
        from TrustVideoLLM.models.share4video.model.builder import load_pretrained_model
        self.tokenizer, self.model, self.processor, self.context_len = load_pretrained_model(model_id, None, model_name, device_map='cpu')
        self.model = self.model.cuda(device).eval()
    

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
        do_sample = generation_kwargs['do_sample']

        
        conv = conv_templates["llava_llama_3"].copy()
    
        qs = DEFAULT_IMAGE_TOKEN + '\n' + user_message
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()


        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device, non_blocking=True)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token is not None else self.tokenizer.eos_token_id

        video_frames = encode_video(video_path, max_frames_num=max_frames_num)
        video_size = video_frames[0].size
        video_tensor = process_images(video_frames, self.processor, self.model.config)[0]


        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=video_tensor.to(
                    dtype=torch.float16, device=self.model.device, non_blocking=True),
                image_sizes=[video_size],
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_token_id,
                use_cache=True,
                )

            response = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True)[0].strip()



        scores = None

        return Response(self.model_id, response, scores, None, extra)
       


