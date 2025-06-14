
from typing import List
from PIL import Image
import torch
from TrustVideoLLM.models.base import BaseChat, Response
from TrustVideoLLM.utils.registry import registry
from modelscope import AutoConfig, AutoModel
from decord import VideoReader, cpu 
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor


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
class mPlug_Owl3(BaseChat):

    model_family = {
        'mPLUG-Owl3-7B':'mPLUG/mPLUG-Owl3-7B-240728'
    }

   
    def __init__(self, model_id: str, device: str = "cuda"):
        super().__init__(model_id)
        self.device = device
        model_id = self.model_family[self.model_id]
      
        self.model = AutoModel.from_pretrained(model_id, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16, trust_remote_code=True)
        self.model.eval().to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.processor = self.model.init_processor(self.tokenizer)
    

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
            

        conversation = [
            {"role": "user", "content": f"""<|video|>
        {user_message}"""},
            {"role": "assistant", "content": ""}
        ]

        max_frames_num = 64
        max_new_tokens = generation_kwargs['max_new_tokens']
        if video_path != None:
            video_frames = [encode_video(video_path, max_frames_num) ]
        else:
            video_frames = None
        inputs = self.processor(conversation, images=None, videos=video_frames).to(self.device)
    
        inputs.update({
            'tokenizer': self.tokenizer,
            'max_new_tokens':max_new_tokens,
            'decode_text':True,
        })

        output = self.model.generate(**inputs)

        scores = None

        return Response(self.model_id, output, scores, None, extra)
       


