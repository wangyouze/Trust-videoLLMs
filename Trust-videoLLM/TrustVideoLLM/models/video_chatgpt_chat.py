import os
from typing import List
from PIL import Image
import torch
from TrustVideoLLM.models.base import BaseChat, Response
from TrustVideoLLM.utils.registry import registry
from TrustVideoLLM.models.video_chatgpt.eval.model_utils import initialize_model, load_video
from TrustVideoLLM.models.video_chatgpt.video_conversation import conv_templates, SeparatorStyle
from TrustVideoLLM.models.video_chatgpt.model.utils import KeywordsStoppingCriteria
from TrustVideoLLM.models.video_chatgpt.constants import *


def get_spatio_temporal_features_torch(features):
    """
    Computes spatio-temporal features from given features.

    Parameters:
    features (torch.Tensor): Input features to process.

    Returns:
    torch.Tensor: Spatio-temporal features.
    """

    # Extract the dimensions of the features
    t, s, c = features.shape

    # Compute temporal tokens as the mean along the time axis
    temporal_tokens = torch.mean(features, dim=1)

    # Padding size calculation
    padding_size = 100 - t

    # Pad temporal tokens if necessary
    if padding_size > 0:
        padding = torch.zeros(padding_size, c, device=features.device)
        temporal_tokens = torch.cat((temporal_tokens, padding), dim=0)

    # Compute spatial tokens as the mean along the spatial axis
    spatial_tokens = torch.mean(features, dim=0)

    # Concatenate temporal and spatial tokens and cast to half precision
    concat_tokens = torch.cat([temporal_tokens, spatial_tokens], dim=0).half()

    return concat_tokens

@registry.register_videoLLM()
class Video_ChatGPT(BaseChat):

    model_family = {
        'Video-ChatGPT-7B': "MBZUAI/Video-ChatGPT-7B"
    }
    def __init__(self, model_id: str, device: str = "cuda:0"):
        super().__init__(model_id)
        self.device = device
        model_id = self.model_family[self.model_id]

        self.conv_mode = "video-chatgpt_v1"

        print("start to load model")
        self.model, self.vision_tower, self.tokenizer, self.image_processor, self.video_token_len = initialize_model(
            os.path.join(model_id, "LLaVA-7B-Lightening-v1-1"),
            os.path.join(model_id, "video_chatgpt-7B.bin")
        )
        print('load completed')
    

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

        qs = user_message + '\n' + DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * self.video_token_len + DEFAULT_VID_END_TOKEN
       

        # Prepare conversation prompt
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize the prompt
        inputs = self.tokenizer([prompt])

        # Preprocess video frames and get image tensor
        video_frames = load_video(video_path, num_frm=max_frames_num)
        image_tensor = self.image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']

        # Move image tensor to GPU and reduce precision to half
        image_tensor = image_tensor.half().cuda(self.device)

        # Generate video spatio-temporal features
        with torch.no_grad():
            image_forward_outs = self.vision_tower(image_tensor, output_hidden_states=True)
            frame_features = image_forward_outs.hidden_states[-2][:, 1:] # Use second to last layer as in LLaVA
        video_spatio_temporal_features = get_spatio_temporal_features_torch(frame_features)

        # Move inputs to GPU
        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        # Define stopping criteria for generation
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        # print("start to inference")
        # Run model inference
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                video_spatio_temporal_features=video_spatio_temporal_features.unsqueeze(0),
                do_sample=do_sample,
                temperature=0.2,
                max_new_tokens=max_new_tokens,
                stopping_criteria=[stopping_criteria])

        # Check if output is the same as input
        n_diff_input_output = (input_ids != output_ids[:, :input_ids.shape[1]]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

        # Decode output tokens
        outputs = self.tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

        # Clean output string
        response = outputs.strip().rstrip(stop_str).strip()


        scores = None

        return Response(self.model_id, response, scores, None, extra)
       


