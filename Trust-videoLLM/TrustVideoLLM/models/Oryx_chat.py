
from typing import List
import torch
from TrustVideoLLM.models.base import BaseChat, Response
from TrustVideoLLM.utils.registry import registry
from decord import VideoReader, cpu
import numpy as np
import transformers
from PIL import Image
import math
from typing import Dict
import re

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids[:2]
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens 
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids 
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids

@registry.register_videoLLM()
class Oryx(BaseChat):

    model_family = {
        "Oryx-1.5-7B": "THUdyh/Oryx-1.5-7B",
        "Oryx-34B": "THUdyh/Oryx-34B"
    }
    

    def __init__(self, model_id: str, device: str = "cuda:0"):
        super().__init__(model_id)
        
        from TrustVideoLLM.models.oryx.model.builder import load_pretrained_model
        from TrustVideoLLM.models.oryx.conversation import conv_templates, SeparatorStyle
        from TrustVideoLLM.models.oryx.utils import disable_torch_init
        from TrustVideoLLM.models.oryx.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from TrustVideoLLM.models.oryx.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_anyres_video_genli
            
        self.device = device

        disable_torch_init()
        self.model_path = self.model_family[self.model_id]
        model_name = get_model_name_from_path(self.model_path)
       
        overwrite_config = {}
        overwrite_config["mm_resampler_type"] = "dynamic_compressor"
        overwrite_config["patchify_video_feature"] = False
        overwrite_config["attn_implementation"] = "sdpa" if torch.__version__ >= "2.1.2" else "eager"

        if '7B' in self.model_path:
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(self.model_path, None, model_name, device_map="cuda:0", overwrite_config=overwrite_config)
        elif '34B' in self.model_path:
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(self.model_path, None, model_name, device_map="auto", overwrite_config=overwrite_config)
        
        self.model.to('cuda').eval()

        self.conv_templates = conv_templates
        self.disable_torch_init = disable_torch_init
        self.IGNORE_INDEX = IGNORE_INDEX
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.tokenizer_image_token = tokenizer_image_token
        self.SeparatorStyle = SeparatorStyle
        self.process_anyres_video_genli = process_anyres_video_genli

    

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
        frames_upbound=64

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        fps = round(vr.get_avg_fps())
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, frames_upbound, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        video = [Image.fromarray(frame) for frame in spare_frames]

        conv_mode = "qwen_1_5"
        
        question = user_message
        question = "<image>\n" + question

        conv = self.conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if '7B' in self.model_path:
            input_ids = preprocess_qwen([{'from': 'human','value': question},{'from': 'gpt','value': None}], self.tokenizer, has_image=True).cuda()
        elif '34B' in self.model_path:
            input_ids = self.tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda:0')

        video_processed = []
        for idx, frame in enumerate(video):
            self.image_processor.do_resize = False
            self.image_processor.do_center_crop = False
            frame = self.process_anyres_video_genli(frame, self.image_processor)

            video_processed.append(frame.unsqueeze(0))
        
        if frame_idx is None:
            frame_idx = np.arange(0, len(video_processed), dtype=int).tolist()
        
        video_processed = torch.cat(video_processed, dim=0).bfloat16().cuda()
        video_processed = (video_processed, video_processed)

        video_data = (video_processed, (384, 384), "video")

        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
       

        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs=input_ids,
                images=video_data[0][0],
                images_highres=video_data[0][1],
                modalities=video_data[2],
                do_sample=True ,
                temperature=0.2,
                top_p=None,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        response = outputs.strip()

        scores = None

        return Response(self.model_id, response, scores, None, extra)
       


