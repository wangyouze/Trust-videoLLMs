
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
class Video_Hallucination(BaseDataset):
    dataset_ids: Sequence[str] = [
        "hallucination",
    ]
  
    def __init__(self,  dataset_id, method_hook: Optional[BaseMethod] = None, temporal_dir=None, semantic_dir=None, object_ralation_dir=None, interaction_dir=None, fact_dir=None, external_factural_dir=None, external_nonfactural_dir=None, video_dir=None):
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)

        self.video_dir = video_dir

        self.data_list = self._prepare_data_list(temporal_dir)
        self.data_list.extend(self._prepare_data_list(semantic_dir))
        self.data_list.extend(self._prepare_data_list(object_ralation_dir))
        self.data_list.extend(self._prepare_data_list(interaction_dir))
        self.data_list.extend(self._prepare_data_list(fact_dir))
        self.data_list.extend(self._prepare_data_list(external_factural_dir))
        self.data_list.extend(self._prepare_data_list(external_nonfactural_dir))


        print('len(dataset)=', len(self.data_list))


    def _prepare_data_list(self, qa_path) -> list:
        paired_qas = json.load(open(qa_path))
        sub_file = qa_path.split('/')[-2]
        data_list = []
        count = 0
        for qa_dct in paired_qas:
            basic = qa_dct["basic"]
            basic_question = f"{basic['question']}\nAnswer the question using 'yes' or 'no'."
            basic_video_path = os.path.join(self.video_dir, sub_file, 'videos', basic["video"])
            basic_answer = basic.get("answer", "")  
            if not os.path.exists(basic_video_path):
                continue

            halluc = qa_dct["hallucination"]
            halluc_question = f"{halluc['question']}\nAnswer the question using 'yes' or 'no'."
            halluc_video_path = os.path.join(self.video_dir, sub_file, 'videos', halluc["video"])
            halluc_answer = halluc.get("answer", "")
            if not os.path.exists(halluc_video_path):
                continue

            data_list.append({
                    "question": basic_question,
                    "video_path": basic_video_path,
                    "answer": 'yes',
                    "type": "basic"
                })

            data_list.append({
                    "question": halluc_question,
                    "video_path": halluc_video_path,
                    "answer": 'no',
                    'type':"hallucination"
            })

            count += 1
            if count > 30:
                break
        return data_list

    
    def __len__(self):
        return len(self.data_list)
    

    def __getitem__(self, idx):
        
        video_path = self.data_list[idx]['video_path']
        print('video_path:', video_path)

        question = self.data_list[idx]['question']
        
        answer = self.data_list[idx]['answer']

        types =  self.data_list[idx]['type']
        video_frames = None


        if self.method_hook:
            return self.method_hook.run(VideoTxtSample( 
            video_frames=video_frames,
            video_path=video_path,
            question=question, 
            answer=answer,
            extra= types,
           
            ))
            
        return VideoTxtSample( 
            video_frames=video_frames,
            video_path=video_path,
            question=question, 
            answer=answer,
            extra=types,
        )