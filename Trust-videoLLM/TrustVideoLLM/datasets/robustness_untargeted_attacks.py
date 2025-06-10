
import os
import io
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
from .base import BaseDataset
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import matplotlib.pyplot as plt
import cv2
import imageio
import random
from TrustVideoLLM.utils.registry import registry
from TrustVideoLLM.methods.base import BaseMethod
from TrustVideoLLM import VideoTxtSample
from TrustVideoLLM.methods.OOD_video_adversarial_attack import KeyFrameAttack


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
    "Character Order": ("character_order.json", "MVBench/video/perception/videos/", "video", False),
    "Counterfactual Inference": ("counterfactual_inference.json", "MVBench/video/clevrer/video_validation/", "video", False),
}

@registry.register_dataset()
class UntargetedAttackDataset(BaseDataset):
    dataset_ids: Sequence[str] = [
        "UntargetedAttackDataset",
        "Clean_untargetedAttackDataset"
    ]
    # dataset_config: Optional[str] = "TrustVideoLLM/configs/robustness/OOD-adversarial-attack.yaml"
    def __init__(self,  dataset_id, method_hook: Optional[BaseMethod] = None, data_dir=None, video_path=None, num_segments=8, resolution=224):
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)
        self.data_list_all = []
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
                self.data_list_all.append({
                    'task_type': k,
                    'prefix': os.path.join(video_path, v_1),
                    'data_type': v[2],
                    'bound': v[3],
                    'data': data
                })
        
        file_path = "./data/robustness/untargeted_adversarial_attack/adversarial_attack_videos.json"
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                self.data_list = json.load(f)
            print("数据已加载:", self.data_list)
        else:
            self.data_list = random.sample(self.data_list_all, 100)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.data_list, f)
            print("数据已保存:", self.data_list)


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

        # self.adversarial_attack = KeyFrameAttack(epsilon=16/255., alpha=1/255, 
        #          temporal_weight=0.2, flow_threshold=2.5, device='cuda')
       
    
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
    
    
    def save_video(self, video_array, output_path, fps=30):
        
        T, H, W, C = video_array.shape
        # OpenCV 使用 BGR 格式，如果输入是 RGB，则需要转换
        if C == 3:
            video_array = video_array[:, :, :, ::-1]  # 将 RGB 转换为 BGR
        # 定义视频编码器（例如 MP4 格式）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
        
        for frame in video_array:
            out.write(frame)
        out.release()




    def read_video(self, video_path):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

   
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
    
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取原始帧率
        if fps <= 0:
            fps = 30  # 如果获取失败，默认设置为30
            print(f"Warning: Could not retrieve FPS for {video_path}, defaulting to 30.")
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break  
            
            # 将帧从 BGR 转换为 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        cap.release()
        video_frames = np.array(frames)

        video_id = video_path.split('/')[-1].split('.')[0]
        adv_video_path = './data/robustness/untargeted_adversarial_attack/adv_video_' + video_id + '.mp4'
        if not os.path.exists(adv_video_path):
            adv_video_frames = self.adversarial_attack.generate_attack(video_frames, caption_target=None, use_sliding_window=False)

            clip = ImageSequenceClip(list(video_frames), fps=fps)
            clip.write_videofile(adv_video_path, codec="libx264")
           
        else:
            adv_video_frames = self.read_video_(adv_video_path)

        return adv_video_frames, adv_video_path
    
    def read_video_(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取原始帧率
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # 视频读取完毕
            
            # 将帧从 BGR 转换为 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        cap.release()
        video_frames = np.array(frames)

        return video_frames,
    


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

        video_id = '/'.join(video_path.split('/')[7:])
        video_path = os.path.join('./data/robustness/MVBench', video_id)
        print('video_path:', video_path)
        
        video_frames, adv_video_path = self.read_video(video_path)

        question, answer = self.qa_template(self.data_list[idx]['data'])

        if self.dataset_id == 'UntargetedAttackDataset':
            video_path = adv_video_path
           



        if self.method_hook:
            return self.method_hook.run(VideoTxtSample( 
            video_frames=video_frames,
            video_path=video_path,
            question=question, 
            answer=answer,
            task_type= self.data_list[idx]['task_type']
            ))
            
        return VideoTxtSample( 
            video_frames=video_frames,
            video_path=video_path,
            question=question, 
            answer=answer,
            task_type=self.data_list[idx]['task_type']
        )