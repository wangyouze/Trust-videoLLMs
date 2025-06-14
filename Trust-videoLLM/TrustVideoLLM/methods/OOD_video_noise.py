
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
import numpy as np
import sys
sys.path.append("/data/home/wangyouze/projects/MulTrust-video/TrustVideoLLM/datasets")
from MVBench_dataset import MVBench_dataset, data_list

sys.path.append('/data/home/wangyouze/projects/MulTrust-video/TrustVideoLLM/models')
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from tqdm import tqdm
from decord import VideoReader, cpu
import copy
import json
import random
from add_noise_2_videos import add_blur, add_color_noise, add_occlusion
from torchvision.transforms import ToPILImage
import torchvision.transforms as T
from video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
from torchvision.transforms.functional import InterpolationMode
import cv2

def get_sinusoid_encoding_table(n_position=784, d_hid=1024, cur_frame=8, ckpt_num_frame=4, pre_n_position=784): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 
    
    # generate checkpoint position embedding
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(pre_n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 
    sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)
    
    print(f"n_position: {n_position}")
    print(f"pre_n_position: {pre_n_position}")
    
    if n_position != pre_n_position:
        T = ckpt_num_frame # checkpoint frame
        P = 14 # checkpoint size
        C = d_hid
        new_P = int((n_position // cur_frame) ** 0.5) # testing size
        if new_P != 14:
            print(f'Pretraining uses 14x14, but current version is {new_P}x{new_P}')
            print(f'Interpolate the position embedding')
            sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
            sinusoid_table = sinusoid_table.reshape(-1, P, P, C).permute(0, 3, 1, 2)
            sinusoid_table = torch.nn.functional.interpolate(
                sinusoid_table, size=(new_P, new_P), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            sinusoid_table = sinusoid_table.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
            sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
    
    if cur_frame != ckpt_num_frame:
        print(f'Pretraining uses 4 frames, but current frame is {cur_frame}')
        print(f'Interpolate the position embedding')
        T = ckpt_num_frame # checkpoint frame
        new_T = cur_frame # testing frame
        # interpolate
        P = int((n_position // cur_frame) ** 0.5) # testing size
        C = d_hid
        sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
        sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)  # BHW, C, T
        sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode='linear')
        sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3) # B, T, H, W, C
        sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
        
    return sinusoid_table

def load_video(video_path, max_frames_num, fps=1,  add_noise=True, force_sample=False):
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

    if add_noise:
        for i in range(len(spare_frames)):
            frame = spare_frames[i]

            # output_path = os.path.join(output_dir, f"frame_{i:04d}.png")  # 文件名格式：frame_0000.png, frame_0001.png, ...
            # cv2.imwrite(output_path, frame)

            # 随机选择一种或多种噪声
            noise_types = random.sample(['blur', 'occlusion', 'color'], random.randint(1, 3))
            for noise_type in noise_types:
                if noise_type == 'blur':
                    frame = add_blur(frame)
                elif noise_type == 'occlusion':
                    frame = add_occlusion(frame)
                elif noise_type == 'color':
                    frame = add_color_noise(frame)
            spare_frames[i] = frame
            # output_path = os.path.join(output_dir, f"noise_frame_{i:04d}.png")  # 文件名格式：frame_0000.png, frame_0001.png, ...
            # cv2.imwrite(output_path, frame)
   
    return spare_frames, frame_time, video_time


def check_ans(pred, gt):
    flag = False
    
    pred_list = pred.lower().split(' ')
    pred_option, pred_content = pred_list[0], ' '.join(pred_list[1:])
    gt_list = gt.lower().split(' ')
    gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])
    if gt_content[-1] == '.':
        gt_content = gt_content[:-1]
    
    if pred_option.replace('.', '') in gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True
        
    return flag



def run_llava_video(args, dataset):

    model_name = args.model_name
    pretrained_model_path = args.pretrained_model
    device = args.device
    
    # 加载模型
    if model_name == "llava_qwen":
        device_map = "auto"
        tokenizer, model, image_processor, _ = load_pretrained_model(
            pretrained_model_path, None, model_name, torch_dtype="bfloat16", device_map=device_map
        )
        model.eval()

    num_frame = args.num_frame
   
    system = "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n"
       


    save_path = args.save_path

    correct = 0
    total = 0
    res_list = []
    acc_dict = {}
    conv_template = "qwen_1_5"

    for example in tqdm(dataset):
        if example == None:
            continue
        task_type = example['task_type']
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0] # correct, total
        acc_dict[task_type][1] += 1
        total += 1


        video_path = example["video_path"]
        if not os.path.exists(video_path):
            print(video_path)
        continue

        # 加载视频
        video, frame_time, video_time = load_video(video_path, num_frame, 1, add_noise=args.add_noise, force_sample=True)
        video_tensor = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].to(device).bfloat16()
        video_tensor = [video_tensor]

        # 构造问题
        time_instruction = (
            f"The video lasts for {video_time:.2f} seconds, and {len(video_tensor[0])} frames are uniformly sampled from it. "
            f"These frames are located at {frame_time}. Please answer the following questions related to this video."
        )
        question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\n" + system + example['question']

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
        pred = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        

        gt = example['answer']
        res_list.append({
            'pred': pred,
            'gt': gt
        })
        if check_ans(pred=pred, gt=gt):
            acc_dict[task_type][0] += 1
            correct += 1
        print(f"Part  Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100 :.2f}%")
        print(f"Total Acc: {correct / total * 100 :.2f}%")
        print('-' * 30, task_type, '-' * 30)

    with open(f"{save_path}.json", "w") as f:
        json.dump({
            "acc_dict": acc_dict,
            "res_list": res_list
        }, f)







if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate captions for videos in a directory using Video-LLM.")
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    parser.add_argument("--data_path", type=str, default="/data1/home/wangyouze/dataset/MultiTrust-video/MVBench/json", help="json file path.")
    parser.add_argument("--video_root", type=str, default="/data1/home/wangyouze/dataset/MultiTrust-video/", help="Path to the directory containing video files.")
    parser.add_argument("--save_path", type=str,default='/data/home/wangyouze/projects/MulTrust-video/test/results/LLaVA-Video-7B-MVBench_add_noise_False', help="Path to save the output JSON file.")
    parser.add_argument("--pretrained_model", type=str, default="/data1/home/wangyouze/checkpoints/LLaVA-Video-7B-Qwen2/", help="Path to the pretrained model directory.")
    parser.add_argument("--num_frame", type=int, default=64, help="Maximum number of frames to sample from each video.")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--add_noise", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda' or 'cpu').")

    args = parser.parse_args()

    dataset = MVBench_dataset(args.data_path, args.video_root, data_list, num_segments=args.num_frame, resolution=args.resolution)

    run_llava_video(args, dataset)
