
import os
import numpy as np
from typing import Optional, Sequence
import torchvision.transforms as T
from .base import BaseDataset
from torchvision import transforms
import pandas as pd
import cv2
from pathlib import Path
import json
from TrustVideoLLM.utils.registry import registry
from TrustVideoLLM.methods.base import BaseMethod
from TrustVideoLLM import VideoTxtSample



@registry.register_dataset()
class VideoEventsUnderstanding(BaseDataset):
    dataset_ids: Sequence[str] = [
        "YouCook2_sampled",
    ]
  
    def __init__(self,  dataset_id, method_hook: Optional[BaseMethod] = None, data_dir=None, video_dir=None):
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)

       
        with open(data_dir, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.data_list = []
        for value in data:
            self.data_list.append({'id': value['id'], 'sentence':value['positive_sentence'], 'answer':'yes'})
            self.data_list.append({'id': value['id'], 'sentence':value['negative_sentence'], 'answer':'no'})
       

        self.video_dir = video_dir

        print('len(dataset)=', len(self.data_list))

    
    def __len__(self):
        return len(self.data_list)
    

    def __getitem__(self, idx):
        
        video_id = self.data_list[idx]['id']
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm']  # 常见视频格式

        video_path = None
        for ext in video_extensions:
            temp_path = os.path.join(self.video_dir, video_id + ext)
            if os.path.exists(temp_path):
                video_path = temp_path
                break
        print('video_path:', video_path)
        if video_path == None:
            print('video_id=', video_id)

        
        question = self.data_list[idx]['sentence'] + "\nGiven a short video clip and a sentence that describes a possible event, please answer 'Yes' if the described event actually happens in the video, otherwise answer 'No'."
        
        answer = self.data_list[idx]['answer']
        video_frames = None


        if self.method_hook:
            return self.method_hook.run(VideoTxtSample( 
            video_frames=video_frames,
            video_path=video_path,
            question=question, 
            answer=answer,
            task_type= 'Dis'
            ))
            
        return VideoTxtSample( 
            video_frames=video_frames,
            video_path=video_path,
            question=question, 
            answer=answer,
            task_type='Dis'
        )