import datetime
import time
import PIL
import os
import sys
import json
import requests

from openai import OpenAI
from .utils import convert_pil_image_to_base64

class LanguageModel():
    def __init__(self, support_vision):
        self._support_vision = support_vision

    def support_vision(self)-> bool:
        return self._support_vision


# class LLaVA(ChatModel):
#     from llava.conversation import (default_conversation, conv_templates,
#                                     SeparatorStyle)
#     from llava.constants import LOGDIR
#     from llava.utils import (build_logger, server_error_msg,
#         violates_moderation, moderation_msg)
#     import hashlib
#
#     def __init__(self,
#                  model_name="llava_llama-2",
#                  server_addr="127.0.0.1:21002"
#                  ):
#         self.model_name = model_name
#         self.server_addr = server_addr

#     def create_conversation(model_name="llava_llama-2"):
#         # First round of conversation
#         # if "llava" in model_name.lower():
#         #     if 'llama-2' in model_name.lower():
#         #         template_name = "llava_llama_2"
#         #     elif "v1" in model_name.lower():
#         #         if 'mmtag' in model_name.lower():
#         #             template_name = "v1_mmtag"
#         #         elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
#         #             template_name = "v1_mmtag"
#         #         else:
#         #             template_name = "llava_v1"
#         #     elif "mpt" in model_name.lower():
#         #         template_name = "mpt"
#         #     else:
#         #         if 'mmtag' in model_name.lower():
#         #             template_name = "v0_mmtag"
#         #         elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
#         #             template_name = "v0_mmtag"
#         #         else:
#         #             template_name = "llava_v0"
#         # elif "mpt" in model_name:
#         #     template_name = "mpt_text"
#         # elif "llama-2" in model_name:
#         #     template_name = "llama_2"
#         # else:
#         #     template_name = "vicuna_v1"
#         template_name = "llava_llama_2"
            
#         # new_state.append_message(new_state.roles[0], chat_history.messages[-2][1])
#         # new_state.append_message(new_state.roles[1], None)
#         chat_history = conv_templates[template_name].copy()
#         return chat_history


#     def generate(self, chat_history, text: str, image: PIL.Image, **kwargs):
#         # Parse configs
#         temperature = kwargs["temperature"]
#         top_p = kwargs["top_p"]
#         max_new_tokens = kwargs["max_new_tokens"]

#         start_tstamp = time.time()
#         if len(text) <= 0 and image is None:
#             raise ValueError("Empty content.")
        
#         text = text[:1536]  # Hard cut-off
#         if image is not None:
#             text = text[:1200]  # Hard cut-off for images
#             if '<image>' not in text:
#                 # text = '<Image><image></Image>' + text
#                 text = text + '\n<image>'
#             # image_process_modes = ["Crop", "Resize", "Pad", "Default"]
#             image_process_mode = "Default"
#             text = (text, image, image_process_mode)
#             # TODO: Only 1 image allowed currently
#             # if len(chat_history.get_images(return_pil=True)) > 0:
#             #     chat_history = default_conversation.copy()
#         chat_history.append_message(chat_history.roles[0], text)
#         chat_history.append_message(chat_history.roles[1], None)
#         chat_history.skip_next = False

#         # Construct prompt  
#         prompt = chat_history.get_prompt()

#         # Make requests
#         pload = {
#             "model": model_name,
#             "prompt": prompt,
#             "temperature": float(temperature),
#             "top_p": float(top_p),
#             "max_new_tokens": min(int(max_new_tokens), 1536),
#             "stop": chat_history.sep if chat_history.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else chat_history.sep2,
#             "images": chat_history.get_images(return_pil=False)
#         }

#         chat_history.messages[-1][-1] = "▌"

#         try:
#             # Stream output
#             response = requests.post(self.server_addr + "/worker_generate_stream",
#                 headers={"User-Agent": "LLaVA Client"}, json=pload, stream=True, timeout=10)
#             for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
#                 if chunk:
#                     data = json.loads(chunk.decode())
#                     if data["error_code"] == 0:
#                         output = data["text"][len(prompt):].strip()
#                         chat_history.messages[-1][-1] = output + "▌"
#                     else:
#                         raise AssertionError(f"Server responds with an error: {data['text']} (error_code: {data['error_code']})")

#                     time.sleep(0.1)
#         except requests.exceptions.RequestException as e:
#             chat_history.messages[-1][-1] = server_error_msg
#             raise IOError(f"Server {server_addr} not reachable.")

#         # Generation completes
#         chat_history.messages[-1][-1] = chat_history.messages[-1][-1][:-1]

#         text_response = chat_history.messages[-1][-1]

#         finish_tstamp = time.time()

#         return text_response, chat_history


class GPT4V(LanguageModel):
    def __init__(self, model="gpt-4-vision-preview"):
        self.model = model

        super().__init__(
            support_vision=True
        )
    def __init__(self, model="gpt-4-vision-preview"):
        self.model = model

        super().__init__(
            support_vision=True
        )

    def chat(self, prompt, image, meta_prompt=""):
        base64_image = convert_pil_image_to_base64(image)

        # Get OpenAI API Key from environment variable
        api_key = os.environ["OPENAI_API_KEY"]
        client = OpenAI(
            api_key=api_key,
        )

        response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": meta_prompt}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        max_tokens=500,
        )
        
        ret = response.choices[0]['message']['content']
        return ret


class GPT4(LanguageModel):
    def __init__(self,):
        self.model = "gpt-4"

        super().__init__(
            support_vision=True
        )

    def chat(self, prompt, meta_prompt=""):
        # Get OpenAI API Key from environment variable
        api_key = os.environ["OPENAI_API_KEY"]


        client = OpenAI(
            api_key=api_key,
        )
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": meta_prompt
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model,
        )
        
        return response.choices[0].message.content
