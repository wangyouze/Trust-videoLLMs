import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import json
import torch
import copy
import warnings
import numpy as np
from decord import VideoReader, cpu
import sys
sys.path.append('/data/home/wangyouze/projects/MulTrust-video/TrustVideoLLM/models')
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from tqdm import tqdm

# 忽略警告
warnings.filterwarnings("ignore")


def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    """
    加载视频并采样固定帧数
    Args:
        video_path (str): 视频文件路径
        max_frames_num (int): 最大采样帧数
        fps (int): 每秒采样帧数
        force_sample (bool): 是否强制均匀采样

    Returns:
        tuple: 视频帧数组、帧时间信息、视频时长
    """
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, frame_time, video_time


def generate_video_captions(video_dir, output_json_path, pretrained_model_path, max_frames_num=64, device="cuda"):
    """
    批量生成视频描述并保存到 JSON 文件中
    Args:
        video_dir (str): 视频目录路径
        output_json_path (str): 输出 JSON 文件路径
        pretrained_model_path (str): 预训练模型路径
        max_frames_num (int): 最大采样帧数
        device (str): 使用的设备（如 'cuda' 或 'cpu'）

    Returns:
        None
    """
    # 加载模型
    model_name = "llava_qwen"
    device_map = "auto"
    tokenizer, model, image_processor, _ = load_pretrained_model(
        pretrained_model_path, None, model_name, torch_dtype="bfloat16", device_map=device_map
    )
    model.eval()

    # 初始化对话模板
    conv_template = "qwen_1_5"

    # 遍历视频目录
    results = {}
    for video_name in tqdm(sorted(os.listdir(video_dir))):

        for name in sorted(os.listdir(os.path.join(video_dir, video_name))):
            video_path = os.path.join(video_dir, video_name, name)

            # 跳过非视频文件
            if not name.endswith(".mp4"):
                continue

            # 加载视频
            video, frame_time, video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
            video_tensor = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].to(device).bfloat16()
            video_tensor = [video_tensor]

            # 构造问题
            time_instruction = (
                f"The video lasts for {video_time:.2f} seconds, and {len(video_tensor[0])} frames are uniformly sampled from it. "
                f"These frames are located at {frame_time}. Please answer the following questions related to this video."
            )
            question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\nPlease describe this video in detail."

            # 构建对话
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            # Tokenize 输入
            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

            # 生成回答
            cont = model.generate(
                input_ids,
                images=video_tensor,
                modalities=["video"],
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
            )
            text_output = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

            # 保存结果
            results[name] = text_output
            print(f"Processed: {name} -> {text_output}")

          

    # 将结果保存为 JSON 文件
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Captions saved to {output_json_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate captions for videos in a directory using Video-LLM.")
    parser.add_argument("--video_dir", type=str, default="/data1/home/wangyouze/dataset/MultiTrust-video/Capera/Videos/Test/", help="Path to the directory containing video files.")
    parser.add_argument("--output_json", type=str,default='/data/home/wangyouze/projects/MulTrust-video/test/results/LLaVA-Video-7B.json', help="Path to save the output JSON file.")
    parser.add_argument("--pretrained_model", type=str, default="/data1/home/wangyouze/checkpoints/LLaVA-Video-7B-Qwen2/", help="Path to the pretrained model directory.")
    parser.add_argument("--max_frames", type=int, default=64, help="Maximum number of frames to sample from each video.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda' or 'cpu').")

    args = parser.parse_args()

    generate_video_captions(
        video_dir=args.video_dir,
        output_json_path=args.output_json,
        pretrained_model_path=args.pretrained_model,
        max_frames_num=args.max_frames,
        device=args.device,
    )
