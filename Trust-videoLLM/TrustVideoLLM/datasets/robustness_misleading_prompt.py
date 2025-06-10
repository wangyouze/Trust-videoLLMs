import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Sequence
from TrustVideoLLM.methods.base import BaseMethod
from TrustVideoLLM.datasets.base import BaseDataset, collate_fn
from TrustVideoLLM.utils.registry import registry
from TrustVideoLLM import VideoTxtSample, TxtSample, _OutputType
import yaml
import os
import json


@registry.register_dataset()
class MisleadingPromptVideos(BaseDataset):
    dataset_ids: Sequence[str] = ["Misleading-Prompt-Videos"]
    dataset_config: Optional[str] = "./TrustVideoLLM/configs/robustness/robustness-misleading-prompt-videos.yaml"
    def __init__(self, dataset_id: str, method_hook: Optional[BaseMethod] = None, **kwargs) -> None:
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)
        with open(self.dataset_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.video_dir = self.config['dataset_cfg'].get('video_dir', '')
        self.annotation_file = self.config['dataset_cfg'].get('annotation_file', '')

        print("video_dir:", self.video_dir)
        print("annotation_file:", self.annotation_file)
       
            
        with open(self.annotation_file, 'r') as f:
            self.data = json.load(f)


        self.dataset = []
        for tid, each_line in enumerate(self.data):
            
            prompt = each_line['misleading prompt']
            label = each_line['Excepted Output']
            video_id = each_line['videoname']
           
                
            video_path = os.path.join(self.video_dir, video_id)
            print('related_video_path:', video_path)
            
            self.dataset.append(VideoTxtSample(video_frames=None, video_path=video_path, question=prompt, answer=label, task_type=None))
            


    def __getitem__(self, index: int) -> _OutputType:

        data = self.dataset[index]

        if self.method_hook:
            return self.method_hook.run(data)
        return data
    
    def __len__(self) -> int:
        return len(self.dataset)
    
