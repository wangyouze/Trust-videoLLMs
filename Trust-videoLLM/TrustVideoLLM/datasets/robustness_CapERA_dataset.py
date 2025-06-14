
import os
import io
import json
import numpy as np
from typing import Optional, Sequence
from decord import VideoReader, cpu
import torchvision.transforms as T
import random
from .base import BaseDataset
from TrustVideoLLM.utils.registry import registry
from TrustVideoLLM.methods.base import BaseMethod
from TrustVideoLLM import VideoTxtSample


@registry.register_dataset()
class CapERA(BaseDataset):
    dataset_ids: Sequence[str] = [
        "CapERA",
    ]

    def __init__(self,  dataset_id="MVBench", method_hook: Optional[BaseMethod] = None, data_dir=None, video_path=None):
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)\
        
        self.video_path = video_path
        
        with open(data_dir, 'r', encoding='utf-8') as file:
            self.data_list = json.load(file)


        print('len(dataset)=', len(self.data_list))
        
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

   
    def __getitem__(self, idx):
        

        video_path = os.path.join(self.video_path, self.data_list[idx]['video_id'].split('.')[0] + ' .mp4')
       
        if not os.path.exists(video_path):
            print(video_path)
            return VideoTxtSample(
            video_frames=None,
            video_path=None,
            question=None, 
            answer=None,
            task_type=None
        )

        question = "Generate a concise caption for the input video. The caption should describe the main event, key objects, location context, actions. Focus on factual and clear language, avoiding speculation or overly creative phrasing. Example: 'person riding a motorcycle passes in front of a car that almost hit him.'"
        answer = self.data_list[idx]['annotation']['English_caption']

        if self.method_hook:
            return self.method_hook.run(VideoTxtSample( 
            video_frames=None,
            video_path=video_path,
            question=question, 
            answer=answer,
            task_type="caption"
            ))
            
        return VideoTxtSample(
            video_frames=None,
            video_path=video_path,
            question=question, 
            answer=answer,
            task_type="caption"
        )