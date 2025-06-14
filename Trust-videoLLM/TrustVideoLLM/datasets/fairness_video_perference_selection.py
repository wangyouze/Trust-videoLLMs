import json
import os
from typing import Optional, Sequence

import yaml
from TrustVideoLLM import VideoTxtSample, _OutputType
from TrustVideoLLM.datasets.base import BaseDataset
from TrustVideoLLM.methods.base import BaseMethod
from TrustVideoLLM.utils.registry import registry
from moviepy import *

@registry.register_dataset()
class VideoPreferenceSelection(BaseDataset):
    dataset_ids: Sequence[str] = ["video-preference-selection"]
    def __init__(self, dataset_id: str, force_prompt: bool = True, method_hook: Optional[BaseMethod] = None, data_dir=None, video_dir=None, **kwargs) -> None:
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)

        self.data_list = []
        with open(data_dir, 'r', encoding='utf-8') as f:
                data_content = json.load(f)['videoPairs']

        for data in data_content:
            output_path = os.path.join(video_dir, str(data['id'])+'.mp4')
            if not os.path.exists(output_path):
                video_path_1 = os.path.join(video_dir, str(data['id'])+'-1.mp4')
                video_path_2 = os.path.join(video_dir, str(data['id'])+'-2.mp4')
                clip_1 = VideoFileClip(video_path_1)
                clip_2 = VideoFileClip(video_path_2)
                
                final_clip = concatenate_videoclips([clip_1, clip_2])
                
                final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
            
                clip_1.close()
                clip_2.close()
                final_clip.close()

            self.data_list.append({'video_path':output_path, 'prompt':data['Prompt']})
        print('len(dataset)=', len(self.data_list))


    def __getitem__(self, index: int) -> _OutputType:
        data = self.data_list[index]
        video_path = data['video_path']
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
    