
import os
import json
import torch
from PIL import Image
import numpy as np
from typing import Optional, Sequence
from decord import VideoReader, cpu
import torchvision.transforms as T
from .video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
from torchvision.transforms.functional import InterpolationMode
# from torch.utils.data import Dataset
from .base import BaseDataset
import cv2
import imageio
import random
from TrustVideoLLM.utils.registry import registry
from TrustVideoLLM.methods.base import BaseMethod
from TrustVideoLLM import VideoTxtSample


data_list = {
    "Action Sequence": ("action_sequence_20250131.json", "MVBench/video/star/Charades_v1_480/", "video", True), # has start & end
    "Action Prediction": ("action_prediction.json", "MVBench/video/star/Charades_v1_480/", "video", True), # has start & end
    "Action Antonym": ("action_antonym.json", "MVBench/video/ssv2_video/", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", "MVBench/video/Moments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", "MVBench/video/FunQA_test/test/", "video", False),
    "Object Existence": ("object_existence_20250131.json", "MVBench/video/clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", "MVBench/video/star/Charades_v1_480/", "video", True), # has start & end
    "Object Shuffle": ("object_shuffle.json", "MVBench/video/perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", "MVBench/videoh/clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", "MVBench/video/sta/sta_video/", "video", True),  # has start & end
    "Scene Transition": ("scene_transition.json", "MVBench/video/scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", "MVBench/video/perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", "MVBench/video/clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", "MVBench/video/clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", "MVBench/video/perception/videos/", "video", False),
    # "Fine-grained Pose": ("fine_grained_pose.json", "MVBench/video/nturgbd/", "video", False),
    "Character Order": ("character_order.json", "MVBench/video/perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", "MVBench/video/vlnqa/", "video", False),
    # "Episodic Reasoning": ("episodic_reasoning.json", "/data1/home/wangyouze/dataset/MultiTrust-video/MVBench/video/tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
    "Counterfactual Inference": ("counterfactual_inference.json", "MVBench/video/clevrer/video_validation/", "video", False),
}

@registry.register_dataset()
class MVBench(BaseDataset):
    dataset_ids: Sequence[str] = [
        "MVBench",
    ]
    dataset_config: Optional[str] = "TrustVideoLLM/configs/robustness/OOD-noise.yaml"
    def __init__(self,  dataset_id="MVBench", method_hook: Optional[BaseMethod] = None, data_dir=None, video_path=None, num_segments=8, resolution=224):
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)
        self.data_list = []
        for k, v in data_list.items():
            v_split = v[1].split('/')
            if v_split[1] == 'videoh':
                v_split[1] = 'video'
                v_1 = '/'.join(v_split)
            else:
                v_1 = v[1]

            with open(os.path.join(data_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': os.path.join(video_path, v_1),
                    'data_type': v[2],
                    'bound': v[3],
                    'data': data
                })
        
        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }
        
        self.num_segments = num_segments
        
        # transform
        crop_size = resolution
        scale_size = resolution
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        self.transform = T.Compose([
            GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(crop_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std) 
        ])
       
    
    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices
    

    def read_video(self, video_path, bound=None):
        # 确保视频路径有效
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            # 初始化视频读取器
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=4)
        except Exception as e:
            raise RuntimeError(f"Error initializing video reader: {e}")

        # 获取视频的最大帧数和FPS
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        # 获取需要读取的帧索引
        images_group = []
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0)

        # 遍历帧索引，读取和处理帧
        for frame_index in frame_indices:
            if frame_index < 0 or frame_index > max_frame:
                print(f"Warning: frame index {frame_index} out of range, skipping.")
                continue

            try:
                # 从 Decord 获取帧并转换为 PIL 图片
                img_array = vr[frame_index].asnumpy()  # 获取帧并转换为 NumPy 数组
                img = Image.fromarray(img_array)  # 转换为 PIL Image

                # 将图像添加到列表中
                images_group.append(img)
            except Exception as e:
                print(f"Error reading frame {frame_index}: {e}")
                continue

        # 如果没有图像，则返回空
        if not images_group:
            raise RuntimeError("No valid frames were read from the video.")

        # 将图像转换为 tensor（假设 `self.transform` 已经定义）
        torch_imgs = self.transform(images_group)

        return torch_imgs

    def read_gif(self, video_path, bound=None, fps=25):
        gif = imageio.get_reader(video_path)
        max_frame = len(gif) - 1
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for index, frame in enumerate(gif):
            if index in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                img = Image.fromarray(img)
                images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs
    
    def read_frame(self, video_path, bound=None, fps=3):
        max_frame = len(os.listdir(video_path))
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=1) # frame_idx starts from 1
        for frame_index in frame_indices:
            img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg"))
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def __getitem__(self, idx):
        
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])

        if not os.path.exists(video_path):
            print(video_path)
            return VideoTxtSample(
            video_path=None,
            question=None, 
            answer=None,
            task_type=None
        )

        question, answer = self.qa_template(self.data_list[idx]['data'])

        if self.method_hook:
            return self.method_hook.run(VideoTxtSample( 
            video_path=video_path,
            question=question, 
            answer=answer,
            task_type= self.data_list[idx]['task_type']
            ))
            
        return VideoTxtSample(
            # 'video': torch_imgs, 
            video_path=video_path,
            question=question, 
            answer=answer,
            task_type=self.data_list[idx]['task_type']
        )