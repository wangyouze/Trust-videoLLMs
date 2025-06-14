
import os
import json
from typing import Optional, Sequence
from .base import BaseDataset
import random
import copy
from TrustVideoLLM.utils.registry import registry
from TrustVideoLLM.methods.base import BaseMethod
from TrustVideoLLM import VideoTxtSample


data_list = {
    "Action Sequence": ("action_sequence_20250131.json", "MVBench/video/star/Charades_v1_480/", "video", True), # has start & end
    "Action Prediction": ("action_prediction.json", "MVBench/video/star/Charades_v1_480/", "video", True), # has start & end
    "Action Antonym": ("action_antonym.json", "MVBench/video/ssv2_video/", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", "MVBench/video/Moments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", "MVBench/video/FunQA_test/test/", "video", False),
    "Object Existence": ("object_existence_20250131.json", "MVBench/video/clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", "MVBench/video/star/Charades_v1_480/", "video", True), # has start & end
    "Object Shuffle": ("object_shuffle.json", "MVBench/video/perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", "MVBench/videoh/clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", "MVBench/video/sta/sta_video/", "video", True),  # has start & end
    "Scene Transition": ("scene_transition.json", "MVBench/video/scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", "MVBench/video/perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", "MVBench/video/clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", "MVBench/video/clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", "MVBench/video/perception/videos/", "video", False),
    "Character Order": ("character_order.json", "MVBench/video/perception/videos/", "video", False),
    "Counterfactual Inference": ("counterfactual_inference.json", "MVBench/video/clevrer/video_validation/", "video", False),
}

@registry.register_dataset()
class AdversarialTextDataset(BaseDataset):
    dataset_ids: Sequence[str] = [
        "AdversarialTextDataset",
    ]
    def __init__(self,  dataset_id, method_hook: Optional[BaseMethod] = None, adv_text_dir=None, data_dir=None, video_path=None, num_segments=8):
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)
        
        
        # file_path = "./data/robustness/adversarial-texts/MVBench_texts_20250527.json"
        if os.path.exists(data_dir):
            with open(data_dir, 'r', encoding='utf-8') as f:
                self.data_list = json.load(f)
            print("Data has been loaded:", self.data_list)

            with open(adv_text_dir, 'r') as f:
                self.adv_texts = json.load(f)

            print('Adv_texts have been loaded:', self.adv_texts)
        else:
            exit('Please check the MVBench data dir')
      

        self.num_segments = num_segments

        self.adv_data_list = []
        for ii, info in enumerate(self.adv_texts):
            video_info = self.data_list[ii]
            if video_info == ' ': 
                continue
            for adv_name, adv_text in info.items():
                if adv_name == 'misleading' or adv_name == 'contradictory': 
                    continue
                
                new_video_info = copy.deepcopy(video_info)
                new_video_info['data']['question'] = adv_text
                new_video_info['adv_name'] = adv_name
                self.adv_data_list.append(new_video_info)

        print('len(dataset)=', len(self.adv_data_list))
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
        return len(self.adv_data_list)
    

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
        
        video_path = os.path.join(self.adv_data_list[idx]['prefix'], self.adv_data_list[idx]['data']['video'])
        video_id = '/'.join(video_path.split('/')[7:])
        video_path = os.path.join('./data/', video_id)
        if os.path.exists(video_path):
            print('video_path:', video_path)
        video_frames = None

        question, answer = self.qa_template(self.adv_data_list[idx]['data']) 
        question = question + '\n Note: Strictly output ONLY the most appropriate option from the given nOptions. Do *not* include any explanations, translations, or additional text'

        if self.method_hook:
            return self.method_hook.run(VideoTxtSample( 
            video_frames=video_frames,
            video_path=video_path,
            question=question, 
            answer=answer,
            extra= self.adv_data_list[idx]['adv_name']
            ))
            
        return VideoTxtSample( 
            video_frames=video_frames,
            video_path=video_path,
            question=question, 
            answer=answer,
            extra=self.adv_data_list[idx]['adv_name']
        )

