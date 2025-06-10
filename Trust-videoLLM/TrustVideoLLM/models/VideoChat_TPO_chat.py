
from typing import List
import torch
from TrustVideoLLM.models.base import BaseChat, Response
from TrustVideoLLM.utils.registry import registry
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@registry.register_videoLLM()
class VideoChat_TPO(BaseChat):

    model_family = {
        /VideoChat-TPO':'OpenGVLab/VideoChat-TPO'
    }

    
    def __init__(self, model_id: str, device: str = "cuda"):
        super().__init__(model_id)
        self.device = device
        model_id = self.model_family[self.model_id]

      
        self.tokenizer =  AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False,)
        self.model = AutoModel.from_pretrained(model_id,  trust_remote_code=True, _tokenizer=self.tokenizer).eval()

      
        

        self.processor = self.build_transform(224)
    
    def build_transform(self, input_size):
        return T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


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

        input_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]



        text = self.processor.apply_chat_template(
                input_messages, tokenize=False, add_generation_prompt=True
            )
        image_inputs, video_inputs, video_kwargs = process_vision_info(input_messages, return_video_kwargs=True)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            fps=max_frames_num,
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
       


