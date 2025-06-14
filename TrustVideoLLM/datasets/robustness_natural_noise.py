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
from .base import BaseDataset
import matplotlib.pyplot as plt
import cv2
import imageio
import random
from TrustVideoLLM.utils.registry import registry
from TrustVideoLLM.methods.base import BaseMethod
from TrustVideoLLM import VideoTxtSample

data_list = {
    "Action Sequence": ("action_sequence_20250131.json", "MVBench/video/star/Charades_v1_480/", "video", True),
    "Action Prediction": ("action_prediction.json", "MVBench/video/star/Charades_v1_480/", "video", True),
    "Action Antonym": ("action_antonym.json", "MVBench/video/ssv2_video/", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", "MVBench/video/Moments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", "MVBench/video/FunQA_test/test/", "video", False),
    "Object Existence": ("object_existence_20250131.json", "MVBench/video/clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", "MVBench/video/star/Charades_v1_480/", "video", True),
    "Object Shuffle": ("object_shuffle.json", "MVBench/video/perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", "MVBench/videoh/clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", "MVBench/video/sta/sta_video/", "video", True),
    "Scene Transition": ("scene_transition.json", "MVBench/video/scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", "MVBench/video/perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", "MVBench/video/clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", "MVBench/video/clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", "MVBench/video/perception/videos/", "video", False),
    "Character Order": ("character_order.json", "MVBench/video/perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", "MVBench/video/vlnqa/", "video", False),
    "Counterfactual Inference": ("counterfactual_inference.json", "MVBench/video/clevrer/video_validation/", "video", False),
}

@registry.register_dataset()
class NaturalNoiseMVBench(BaseDataset):
    dataset_ids: Sequence[str] = ["NaturalNoiseMVBench", "Clean_MVBench"]
    dataset_config: Optional[str] = "TrustVideoLLM/configs/robustness/OOD-noise.yaml"
    
    def __init__(self, dataset_id="MVBench", method_hook: Optional[BaseMethod] = None, 
                 data_dir=None, video_path=None, num_segments=8, resolution=224,
                 noise_prob=0.3, noise_type='mixed'):
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)
        
        error_videos = ['56901.webm', '108710.webm', '8529.webm', '175746.webm', '207257.webm',
                         '159894.webm', '165379.webm', '160267.webm', '220346.webm']
        file_path = "./data/robustness/OOD-noise/natural_noise_videos.json"
        self.data_list = []
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                self.data_list_ = json.load(f)
                for info in self.data_list_:
                    if info['data']['video'] in error_videos:
                        continue
                    self.data_list.append(info)
            print("Sample data has been loaded", self.data_list)

        else:
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
            
            
            self.data_list = random.sample(self.data_list, 200)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.data_list, f)
            print("Sample data has been saved:", self.data_list)
        
        # self.decord_method = {
        #     'video': self.read_video,
        #     'gif': self.read_gif,
        #     'frame': self.read_frame,
        # }
        
        self.num_segments = num_segments
        self.noise_prob = noise_prob  # Probability of applying noise to a frame
        self.noise_type = noise_type  # 'gaussian', 'salt_pepper', or 'mixed'
        
        # Transform
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
    
    def add_gaussian_noise(self, img_array, mean=0, sigma=25):
        """Add Gaussian noise to an image array."""
        noise = np.random.normal(mean, sigma, img_array.shape)
        noisy_img = img_array + noise
        return np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    def add_salt_pepper_noise(self, img_array, salt_prob=0.01, pepper_prob=0.01):
        """Add salt and pepper noise to an image array."""
        noisy_img = img_array.copy()
        # Salt noise (white)
        num_salt = np.ceil(salt_prob * img_array.size)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_array.shape]
        noisy_img[tuple(coords)] = 255
        # Pepper noise (black)
        num_pepper = np.ceil(pepper_prob * img_array.size)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_array.shape]
        noisy_img[tuple(coords)] = 0
        return noisy_img

    def apply_noise(self, img_array):
        """Apply noise to an image array based on noise_type."""
        if self.noise_type == 'gaussian':
            return self.add_gaussian_noise(img_array)
        elif self.noise_type == 'salt_pepper':
            return self.add_salt_pepper_noise(img_array)
        elif self.noise_type == 'mixed':
            if random.random() < 0.5:
                return self.add_gaussian_noise(img_array)
            else:
                return self.add_salt_pepper_noise(img_array)
        return img_array
    def read_video(self, video_path, save_path, bound=None):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=4)
        except Exception as e:
            raise RuntimeError(f"Error initializing video reader: {e}")

        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        images_group = []
        noisy_frames = []  # 用于保存添加噪声后的帧
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0)

        for frame_index in frame_indices:
            if frame_index < 0 or frame_index > max_frame:
                print(f"Warning: frame index {frame_index} out of range, skipping.")
                continue

            try:
                img_array = vr[frame_index].asnumpy()
                # 随机决定是否添加噪声并保存帧
                if random.random() < self.noise_prob:
                    img_array = self.apply_noise(img_array)
                noisy_frames.append(img_array)  # 保存帧到列表
                img = Image.fromarray(img_array)
                images_group.append(img)
            except Exception as e:
                print(f"Error reading frame {frame_index}: {e}")
                continue

        if not images_group:
            raise RuntimeError("No valid frames were read from the video.")

        ext = os.path.splitext(save_path)[1].lower()
        if ext == '.webm':
            codec = 'libvpx'  # 或 'libvpx-vp9' 用于 VP9
        else:
            codec = 'libx264'  # 默认使用 H.264，适用于 .mp4 等
        
        try:
            writer = imageio.get_writer(save_path, fps=fps, codec=codec)
            for frame in noisy_frames:
                writer.append_data(frame)
            writer.close()
            print(f"Noisy video saved to: {save_path}")
        except Exception as e:
            print(f"Error saving noisy video: {e}")

        # torch_imgs = self.transform(images_group)
        # return torch_imgs

    def read_gif(self, video_path, bound=None, fps=25):
        gif = imageio.get_reader(video_path)
        max_frame = len(gif) - 1
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for index, frame in enumerate(gif):
            if index in frame_indices:
                img_array = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                if random.random() < self.noise_prob:
                    img_array = self.apply_noise(img_array)
                img = Image.fromarray(img_array)
                images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs
    
    def read_frame(self, video_path, bound=None, fps=3):
        max_frame = len(os.listdir(video_path))
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=1)
        for frame_index in frame_indices:
            img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg"))
            img_array = np.array(img)
            if random.random() < self.noise_prob:
                img_array = self.apply_noise(img_array)
            img = Image.fromarray(img_array)
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

        if self.dataset_id != 'Clean_MVBench':
            video_name = self.data_list[idx]['data']['video'].split('/')[-1].split('.')[0]
            
            noise_video_path = './data/robustness/OOD-noise/natural_noise_videos/' + video_name + '.mp4'
        
            if not os.path.exists(noise_video_path):
                self.read_video(video_path, noise_video_path)
            
            video_path = noise_video_path
        

        question, answer = self.qa_template(self.data_list[idx]['data'])

        if self.method_hook:
            return self.method_hook.run(VideoTxtSample( 
                video_frames=None,
                video_path=video_path,
                question=question, 
                answer=answer,
                task_type=self.data_list[idx]['task_type']
            ))
            
        return VideoTxtSample(
            video_frames=None,
            video_path=video_path,
            question=question, 
            answer=answer,
            task_type=self.data_list[idx]['task_type']
        )