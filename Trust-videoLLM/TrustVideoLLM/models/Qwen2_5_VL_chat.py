
from typing import List
import torch
from TrustVideoLLM.models.base import BaseChat, Response
from TrustVideoLLM.utils.registry import registry



@registry.register_videoLLM()
class Qwen2_5_VL(BaseChat):
   

    model_family = {
        'Qwen2.5-VL-7B-Instruct':'Qwen/Qwen2.5-VL-7B-Instruct',
         'Qwen2.5-VL-72B-Instruct': 'Qwen/Qwen2.5-VL-72B-Instruct'
    }

   
    def __init__(self, model_id: str, device: str = "cuda"):
        super().__init__(model_id)
        self.device = device
        model_id = self.model_family[self.model_id]

        from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        from qwen_vl_utils import process_vision_info
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
        )

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.process_vision_info = process_vision_info

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
                        user_prompt = message["content"]["question"]
                        extra = message["content"]["extra"]
                        
                    else:
                        user_prompt = message["content"]
                        
                elif message["role"] == "assistant":
                    # TODO: add assistant answer into the conversation
                    pass
            else:
                raise ValueError(
                    "Unsupported role. Only system, user and assistant are supported."
                )
            
        max_new_tokens = generation_kwargs['max_new_tokens']

        input_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 224 * 224,
                        "fps": 1.0,
                        "max_frames": 64,
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]


        print('video_path=', video_path)
        text = self.processor.apply_chat_template(
                input_messages, tokenize=False, add_generation_prompt=True
            )
        image_inputs, video_inputs, video_kwargs = self.process_vision_info(input_messages, return_video_kwargs=True)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )


        scores = None

        return Response(self.model_id, output_text, scores, None, extra)
       


