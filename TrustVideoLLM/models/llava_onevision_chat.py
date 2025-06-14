
from typing import List
import torch
from TrustVideoLLM.models.base import BaseChat, Response
from TrustVideoLLM.utils.registry import registry

@registry.register_videoLLM()
class Llava_OneVision(BaseChat):

    model_family = {
        "llava-onevision-qwen2-72b-ov-hf":"llava-hf/llava-onevision-qwen2-72b-ov-hf",
    }

    def __init__(self, model_id: str, device: str = "cuda"):
        super().__init__(model_id)
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
        self.device = device
       
        model_id = self.model_family[model_id]
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
        ).to(0)

        self.processor = AutoProcessor.from_pretrained(model_id)

    

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
        do_sample = generation_kwargs['do_sample']
        
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "url": video_path},
                    {"type": "text", "text": user_message},
                    ],
            }
        ]


        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        response = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
       
     
        scores = None

        return Response(self.model_id, response, scores, None, extra)
       



