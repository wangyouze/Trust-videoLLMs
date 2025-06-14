
from typing import List
import torch
from TrustVideoLLM.models.base import BaseChat, Response
from TrustVideoLLM.utils.registry import registry
import functools, torch, os, tqdm
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
apply_liger_kernel_to_qwen2_vl() # important. our model is trained with this. keep consistency
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, LogitsProcessor, logging
from livecc_utils import prepare_multiturn_multimodal_inputs_for_generation, get_smart_resized_clip, get_smart_resized_video_reader
from qwen_vl_utils import process_vision_info




@registry.register_videoLLM()
class LiveCC(BaseChat):

    model_family = {
        "LiveCC-7B-Instruct": "chenjoya/LiveCC-7B-Instruct"
    }
    def __init__(self, model_id: str, device: str = "cuda:0"):
        super().__init__(model_id)
        self.device = device
       
        model_id = self.model_family[self.model_id]

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
          model_id, torch_dtype="auto", 
          device_map='cuda', 
          attn_implementation='flash_attention_2'
        )

        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
        self.model.prepare_inputs_for_generation = functools.partial(prepare_multiturn_multimodal_inputs_for_generation, self.model)
       
    

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
        repetition_penalty = 1.05
        max_frames_num = 64

        conversation = []
     
        content = [{"type": "text", "text": user_message}]
        content.insert(0, {"type": "video", "video": video_path, "total_pixels":max_frames_num*768})
        conversation.append({"role": "user", "content": content})

        image_inputs, video_inputs = process_vision_info(conversation)
        texts = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, return_tensors='pt')
        
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            return_attention_mask=False
        )
        inputs.to(self.model.device)
       

        outputs = self.model.generate(
            **inputs, 
            past_key_values=None, 
            return_dict_in_generate=True, 
            do_sample=do_sample, 
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )

        response = self.processor.decode(outputs.sequences[0, inputs.input_ids.size(1):], skip_special_tokens=True)

        scores = None

        return Response(self.model_id, response, scores, None, extra)
       


