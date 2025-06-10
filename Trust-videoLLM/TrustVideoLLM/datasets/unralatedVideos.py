
# This code references MultiTrust
import os
from glob import glob
from typing import Optional, Sequence

import yaml
from natsort import natsorted
from TrustVideoLLM import VideoTxtSample, _OutputType
from TrustVideoLLM.datasets.base import BaseDataset, collate_fn
from TrustVideoLLM.methods.base import BaseMethod
from TrustVideoLLM.utils.registry import registry


@registry.register_dataset()
class UnrelatedVideoDataset(BaseDataset):
    dataset_ids: Sequence[str] = ["unrelated-video-color", "unrelated-video-natural", "unrelated-video-noise"]
    dataset_config: Optional[str] = "TrustVideoLLM/configs/fairness/unrelatedvideo.yaml"

    def __init__(self, dataset_id: str, method_hook: Optional[BaseMethod] = None, **kwargs) -> None:
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)
        with open(self.dataset_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.video_dir = self.config.get('video_dir', '')


        data_type = dataset_id.split('-')[-1]
        self.dataset = [VideoTxtSample(video_path=path, video_frames=None) for path in natsorted(glob(os.path.join(self.video_dir, f'*{data_type}*')))]

    def __getitem__(self, index: int) -> _OutputType:
        if self.method_hook:
            return self.method_hook.run(self.dataset[index])
        return self.dataset[index]
    
    def __len__(self) -> int:
        return len(self.dataset)
    
