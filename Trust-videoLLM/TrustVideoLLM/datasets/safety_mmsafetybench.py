import os
from typing import Optional, Sequence
from TrustVideoLLM.methods.base import BaseMethod
from TrustVideoLLM.datasets.base import BaseDataset, collate_fn
from TrustVideoLLM.utils.registry import registry
from TrustVideoLLM import VideoTxtSample, TxtSample, _OutputType
import cv2
import json
import numpy as np




@registry.register_dataset()
class MMSafetyBenchDataset(BaseDataset):
    
    dataset_ids: Sequence[str] = ["MM-SafetyBench"]
   
    def __init__(self, dataset_id: str, method_hook: Optional[BaseMethod] = None, data_dir=None, video_dir=None, **kwargs) -> None:
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)
        
        self.video_dir = video_dir
        self.annotations = []
        data_infos = json.load(open(data_dir, "r"))
        for data_idx in data_infos.keys():
            data_info = data_infos[data_idx]
                
            self.annotations.append(
                {
                    "question": data_info["Question"],
                    "prompt": data_info["Prompt"],
                    "video_path": os.path.join(self.video_dir, data_info["Video"]),
                }
            )

    
        print(f"{len(self.annotations)} data loaded")


    def __getitem__(self, index: int) -> _OutputType:

        
        video_path = self.annotations[index]['video_path']
        video_frames = None

        question = self.annotations[index]['prompt']
        answer = None

        if self.method_hook:
            return self.method_hook.run(VideoTxtSample( 
            video_frames=video_frames,
            video_path=video_path,
            question=question, 
            answer=answer,
            task_type="jailbreak"
            ))
            
        return VideoTxtSample(
            video_frames=video_frames,
            video_path=video_path,
            question=question, 
            answer=answer,
            task_type="jailbreak"
        )
    
    def __len__(self) -> int:
        return len(self.annotations)
    
   

