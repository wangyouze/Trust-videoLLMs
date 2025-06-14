
from typing import Optional, Sequence, Dict
from TrustVideoLLM.datasets.base import BaseDataset
from TrustVideoLLM.methods.base import BaseMethod
from TrustVideoLLM.utils.registry import registry
from TrustVideoLLM import VideoTxtSample, TxtSample, _OutputType
import json
import os


@registry.register_dataset()
class StereotypicalGenerationDataset(BaseDataset):
    
    dataset_ids: Sequence[str] = ["StereotypicalGenerationDataset"]

    def __init__(self, dataset_id: str, method_hook: Optional[BaseMethod] = None, data_dir=None, video_dir=None,) -> None:
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)

        self.data_list = []
        self.video_dir = video_dir

        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(data_dir, filename)
             
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.data_list.extend(data)
        print('len(dataset)=', len(self.data_list))
        

    def __getitem__(self, index: int) -> _OutputType:

        data = self.data_list[index]
        video_path = os.path.join(self.video_dir, data['video']) 
        video_frames = None
        answer = None

        question = data['prompt']

        if self.method_hook:
            return self.method_hook.run(VideoTxtSample( 
            video_frames=video_frames,
            video_path=video_path,
            question=question, 
            answer=answer,
            task_type= 'Gen'
            ))
            
        return VideoTxtSample( 
            video_frames=video_frames,
            video_path=video_path,
            question=question, 
            answer=answer,
            task_type='Gen'
        )
    
    def __len__(self) -> int:
        return len(self.data_list)
    