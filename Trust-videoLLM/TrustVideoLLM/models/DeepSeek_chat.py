from typing import List, Dict, Any, Literal
import yaml
from TrustVideoLLM.utils.registry import registry
from TrustVideoLLM.models.base import BaseChat, Response
from TrustVideoLLM.utils.utils import get_abs_path
import os
import time
from openai import OpenAI


@registry.register_videoLLM()
class DeepSeekChat(BaseChat):

    MODEL_CONFIG = {
        "deepseek-chat": "configs/models/deepseek.yaml",
    }

    model_family = list(MODEL_CONFIG.keys())

    def __init__(self, model_id: str = "deepseek-chat", **kargs):
    
        # Please install OpenAI SDK first: `pip3 install openai`
        self.model_id = model_id
        config = self.MODEL_CONFIG[self.model_id]
        with open(get_abs_path(config)) as f:
            self.model_config = yaml.load(f, Loader=yaml.FullLoader)

        
        self.api_key = os.getenv("DeepSeek_apikey", "")
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")

        self.max_retries = self.model_config.get("max_retries", 10)
        self.timeout = self.model_config.get("timeout", 1)

    def chat(self, messages: List, **generation_kwargs):

        assert len(messages) == 1, "Only support one-turn conversation currently"
        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:
                if message["role"] == "user":
                    if isinstance(message["content"], dict):
                        # multimodal
                        user_message = message["content"]["question"]
                        
                    else:
                        user_message = message["content"]
                        
                elif message["role"] == "assistant":
                    # TODO: add assistant answer into the conversation
                    pass
            else:
                raise ValueError(
                    "Unsupported role. Only system, user and assistant are supported."
                )
            
        for i in range(self.max_retries):
            try:

                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": user_message},
                    ],
                    stream=False
                )
                break
            except Exception as e:
                print(f"Error in generation: {e}")
                response = f"Error in generation: {e}"
                time.sleep(self.timeout)


        response_message = response.choices[0].message.content
        # print(response_message)
        finish_reason = response.choices[0].finish_reason
        logprobs = response.choices[0].logprobs

        return Response(self.model_id, response_message, logprobs, finish_reason, None)

