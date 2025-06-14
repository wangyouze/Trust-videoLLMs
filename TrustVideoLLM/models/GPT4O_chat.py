from typing import List, Dict, Any, Literal
import openai
from TrustVideoLLM.utils.registry import registry
from TrustVideoLLM.models.base import BaseChat, Response
import base64
import cv2
import os
import time

@registry.register_videoLLM()
class OpenAIChat(BaseChat):
    """
    Chat class for OpenAI models, e.g., gpt-4o
    """

    MODEL_CONFIG = {
        "gpt-4o": "configs/models/openai/openai.yaml",
    }

    model_family = list(MODEL_CONFIG.keys())

    def __init__(self, model_id: str = "gpt-4o", **kwargs):
        super().__init__(model_id=model_id)
        self.model_id = model_id
        self.model = openai.OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL", ""),
            api_key=os.getenv("OPENAI_API_KEY", "")
        )

    def chat(self, messages: List[Dict[str, Any]], **generation_kwargs) -> Response:
        assert len(messages) == 1, "Only support one-turn conversation currently"
        
        for message in messages:
            if message["role"] not in ["system", "user", "assistant"]:
                raise ValueError("Unsupported role. Only system, user, and assistant are supported.")
            if message["role"] == "user":
                if isinstance(message["content"], dict):
                    video_path = message["content"].get("video_path")
                    user_message = message["content"].get("question")
                    extra = message["content"].get("extra")
                    if not video_path or not user_message:
                        raise ValueError("Missing required fields: video_path or question")
                else:
                    user_message = message["content"]
            elif message["role"] == "assistant":
                # TODO: Handle assistant messages for multi-turn conversation
                pass

        max_new_tokens = generation_kwargs.get("max_new_tokens", 512)
        do_sample = generation_kwargs.get("do_sample", True)
        max_frames = 64
        # Process video frames
        base64Frames = []
      
        with cv2.VideoCapture(video_path) as video:
            if not video.isOpened():
                raise ValueError("Failed to open video file")
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_jump = max(1, int(fps))
            frame_count = 0

            while video.isOpened() and len(base64Frames) < max_frames:
                success, frame = video.read()
                if not success:
                    break
                if frame_count % frame_jump == 0:
                    # No resizing, use original frame
                    _, buffer = cv2.imencode(".jpg", frame)
                    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
                frame_count += 1

        # Prepare prompt
        PROMPT_MESSAGES = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": 'These are frames of a video. ' + user_message},
                    *[
                        {
                            "type": "input_image",
                            "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
                        }
                        for frame in base64Frames
                    ],
                ],
            }
        ]

        # API call with retry logic
        params = {
            "model": "chatgpt-4o-latest",
            "messages": PROMPT_MESSAGES,
            "max_tokens": max_new_tokens,
            "temperature": 0.7 if do_sample else 0.0,
            "logprobs": True
        }
        
        max_retries = 5
        retry_delay = 2  # seconds
        for attempt in range(max_retries):
            try:
                result = self.model.chat.completions.create(**params)
                response = result.choices[0].message.content
                scores = result.choices[0].logprobs
                return Response(self.model_id, response, scores, None, extra)
            except openai.OpenAIError as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    raise RuntimeError(f"OpenAI API request failed after {max_retries} attempts: {e}")