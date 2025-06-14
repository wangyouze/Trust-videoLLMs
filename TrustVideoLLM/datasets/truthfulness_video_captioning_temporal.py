
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
class Video_Captioning_TempCompass(BaseDataset):
    dataset_ids: Sequence[str] = [
        "TempCompass",
    ]
  
    def __init__(self,  dataset_id, method_hook: Optional[BaseMethod] = None, data_dir=None, video_dir=None):
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)

        df = pd.read_csv(data_dir)
        self.data_list = df.to_dict('records')
       

        self.video_dir = video_dir

        print('len(dataset)=', len(self.data_list))

    
    def __len__(self):
        return len(self.data_list)
    

    def __getitem__(self, idx):
        
        video_id = self.data_list[idx]['video_id']
        video_path = os.path.join(self.video_dir, video_id+'.mp4')
        print('video_path:', video_path)

        
        question = self.data_list[idx]['question']
        
        answer = {'mc_question': self.data_list[idx]['mc_question'], 'mc_answer': self.data_list[idx]['mc_question']}
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


        