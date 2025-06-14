
# import os
# import json
# import numpy as np
# from typing import Optional, Sequence
# from decord import VideoReader, cpu
# import torchvision.transforms as T
# from .base import BaseDataset
# import cv2
# import random
# from TrustVideoLLM.utils.registry import registry
# from TrustVideoLLM.methods.base import BaseMethod
# from TrustVideoLLM import VideoTxtSample


# data_list = {
#     "Action Sequence": ("action_sequence_20250131.json", "MVBench/video/star/Charades_v1_480/", "video", True), # has start & end
#     "Action Prediction": ("action_prediction.json", "MVBench/video/star/Charades_v1_480/", "video", True), # has start & end
#     "Action Antonym": ("action_antonym.json", "MVBench/video/ssv2_video/", "video", False),
#     "Fine-grained Action": ("fine_grained_action.json", "MVBench/video/Moments_in_Time_Raw/videos/", "video", False),
#     "Unexpected Action": ("unexpected_action.json", "MVBench/video/FunQA_test/test/", "video", False),
#     "Object Existence": ("object_existence_20250131.json", "MVBench/video/clevrer/video_validation/", "video", False),
#     "Object Interaction": ("object_interaction.json", "MVBench/video/star/Charades_v1_480/", "video", True), # has start & end
#     "Object Shuffle": ("object_shuffle.json", "MVBench/video/perception/videos/", "video", False),
#     "Moving Direction": ("moving_direction.json", "MVBench/videoh/clevrer/video_validation/", "video", False),
#     "Action Localization": ("action_localization.json", "MVBench/video/sta/sta_video/", "video", True),  # has start & end
#     "Scene Transition": ("scene_transition.json", "MVBench/video/scene_qa/video/", "video", False),
#     "Action Count": ("action_count.json", "MVBench/video/perception/videos/", "video", False),
#     "Moving Count": ("moving_count.json", "MVBench/video/clevrer/video_validation/", "video", False),
#     "Moving Attribute": ("moving_attribute.json", "MVBench/video/clevrer/video_validation/", "video", False),
#     "State Change": ("state_change.json", "MVBench/video/perception/videos/", "video", False),
#     "Character Order": ("character_order.json", "MVBench/video/perception/videos/", "video", False),
#     "Counterfactual Inference": ("counterfactual_inference.json", "MVBench/video/clevrer/video_validation/", "video", False),
# }

# @registry.register_dataset()
# class TemporalConsistencyDataset(BaseDataset):
#     dataset_ids: Sequence[str] = [
#         "TemporalConsistencyDataset",
#         "OriginDataset"
#     ]
   
#     def __init__(self,  dataset_id, method_hook: Optional[BaseMethod] = None, data_dir=None, video_dir=None):
#         super().__init__(dataset_id=dataset_id, method_hook=method_hook)
#         self.data_list_all = []
#         for k, v in data_list.items():
#             v_split = v[1].split('/')
#             if v_split[1] == 'videoh':
#                 v_split[1] = 'video'
#                 v_1 = '/'.join(v_split)
#             else:
#                 v_1 = v[1]

#             with open(os.path.join(data_dir, v[0]), 'r') as f:
#                 json_data = json.load(f)
#             for data in json_data:
#                 self.data_list_all.append({
#                     'task_type': k,
#                     'prefix': os.path.join(video_dir, v_1),
#                     'data_type': v[2],
#                     'bound': v[3],
#                     'data': data
#                 })
        
#         file_path = "./data/robustness/temporal_consistency/ood_temporal_consistency_videos.json"
#         if os.path.exists(file_path):
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 self.data_list = json.load(f)
#             print("数据已加载:", self.data_list)
#         else:
#             self.data_list = random.sample(self.data_list_all, 200)
#             with open(file_path, 'w', encoding='utf-8') as f:
#                 json.dump(self.data_list, f)
#             print("数据已保存:", self.data_list)

#         self.output_path = './data/robustness/temporal_consistency/temporal_videos/'

#         self.new_data_list = []
#         for idx in range(len(self.data_list)):
#             video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
#             print('video_path:', video_path)

#             video_id = video_path.split('/')[-1].split('.')[0]
#             video_drop_path = os.path.join(self.output_path, video_id + '_drop.mp4')
#             video_shuffle_path = os.path.join(self.output_path, video_id + '_shuffle.mp4')

#             if not os.path.exists(video_drop_path):
#                 self.process_video_drop(video_path, output_path_drop=video_drop_path, drop_ratio=0.2)
#             self.new_data_list.append({'origin_video_path':video_path,
#                                         'video_path': video_drop_path, 
#                                        'data':self.data_list[idx]['data'],
#                                        'task_type':self.data_list[idx]['task_type']})

#             if not os.path.exists(video_shuffle_path):
#                 self.process_video_shuffle(video_path, output_path_shuffle=video_shuffle_path, drop_ratio=0.2)
#             self.new_data_list.append({'origin_video_path':video_path,
#                                         'video_path': video_shuffle_path, 
#                                        'data':self.data_list[idx]['data'],
#                                        'task_type': self.data_list[idx]['task_type']})

        
#     def __len__(self):
#         return len(self.new_data_list)
    
#     def process_video_drop(self, video_path, output_path_drop, drop_ratio=0.2):
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             print("错误：无法打开视频文件")
#             return
        
#         # 获取视频属性
#         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
#         frames = []
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frames.append(frame)
#         cap.release()
        
       
#         total_frames = len(frames)
#         print(f"原始帧数: {total_frames}")
        
#         # 随机丢弃帧
#         drop_num = int(total_frames * drop_ratio) 
#         keep_indices = sorted(random.sample(range(total_frames), total_frames - drop_num))  # 保留的帧索引
#         dropped_frames = [frames[i] for i in keep_indices]
        
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
#         out_drop = cv2.VideoWriter(output_path_drop, fourcc, fps, (width, height))
#         for frame in dropped_frames:
#             out_drop.write(frame)
#         out_drop.release()
#         print(f"丢弃帧后视频已保存到: {output_path_drop}, 帧数: {len(dropped_frames)}")
        

#     def process_video_shuffle(self, video_path, output_path_shuffle, drop_ratio=0.2):
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             print("错误：无法打开视频文件")
#             return
        
#         # 获取视频属性
#         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
#         frames = []
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frames.append(frame)
#         cap.release()
        
       
#         total_frames = len(frames)
#         print(f"原始帧数: {total_frames}")
        
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
       
#         # 打乱帧顺序
#         shuffled_frames = frames.copy()
#         random.shuffle(shuffled_frames) 
        
#         # 保存打乱帧顺序后的视频
#         out_shuffle = cv2.VideoWriter(output_path_shuffle, fourcc, fps, (width, height))
#         for frame in shuffled_frames:
#             out_shuffle.write(frame)
#         out_shuffle.release()
#         print(f"打乱帧顺序后视频已保存到: {output_path_shuffle}, 帧数: {len(shuffled_frames)}")


#     def qa_template(self, data):
#         question = f"Question: {data['question']}\n"
#         question += "Options:\n"
#         answer = data['answer']
#         answer_idx = -1
#         for idx, c in enumerate(data['candidates']):
#             question += f"({chr(ord('A') + idx)}) {c}\n"
#             if c == answer:
#                 answer_idx = idx
#         question = question.rstrip()
#         answer = f"({chr(ord('A') + answer_idx)}) {answer}"
#         return question, answer

#     def __getitem__(self, idx):
        
#         question, answer = self.qa_template(self.new_data_list[idx]['data'])

#         if self.dataset_id == "TemporalConsistencyDataset":
#             video_path = self.new_data_list[idx]['video_path']
#         else:
#             video_path = self.new_data_list[idx]['origin_video_path']

#         if self.method_hook:
#             return self.method_hook.run(VideoTxtSample( 
#             video_frames=None,
#             video_path=video_path,
#             question=question, 
#             answer=answer,
#             task_type= self.new_data_list[idx]['task_type']
#             ))
            
#         return VideoTxtSample( 
#             video_frames=None,
#             video_path=video_path,
#             question=question, 
#             answer=answer,
#             task_type=self.new_data_list[idx]['task_type']
#         )



import os
import json
import numpy as np
from typing import Optional, Sequence
from decord import VideoReader, cpu
import torchvision.transforms as T
from .base import BaseDataset
import cv2
import random
from collections import defaultdict
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
class TemporalConsistencyDataset(BaseDataset):
    dataset_ids: Sequence[str] = [
        "TemporalConsistencyDataset",
        "OriginDataset"
    ]
   
    def __init__(self,  dataset_id, method_hook: Optional[BaseMethod] = None, data_dir=None, video_dir=None):
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)
        self.data_list_all = []
        for k, v in data_list.items():
            v_split = v[1].split('/')
            if v_split[1] == 'videoh':
                v_split[1] = 'video'
                v_1 = '/'.join(v_split)
            else:
                v_1 = v[1]

            with open(os.path.join(data_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list_all.append({
                    'task_type': k,
                    'prefix': os.path.join(video_dir, v_1),
                    'data_type': v[2],
                    'bound': v[3],
                    'data': data
                })
        
        file_path = "./data/robustness/temporal_consistency/ood_temporal_consistency_videos_new.json"
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                self.data_list = json.load(f)
            print("数据已加载:", len(self.data_list))
        else:
            # 按照task_type分组
            task_type_groups = defaultdict(list)
            for item in self.data_list_all:
                task_type_groups[item['task_type']].append(item)
            
            # 每种类型采样20个
            self.data_list = []
            samples_per_type = 20
            
            for task_type, items in task_type_groups.items():
                if len(items) >= samples_per_type:
                    # 如果该类型的数据量够，随机采样20个
                    sampled_items = random.sample(items, samples_per_type)
                else:
                    # 如果该类型的数据量不够，全部取出
                    sampled_items = items
                    print(f"警告: {task_type} 类型只有 {len(items)} 个样本，少于要求的 {samples_per_type} 个")
                
                self.data_list.extend(sampled_items)
                print(f"{task_type}: 采样了 {len(sampled_items)} 个样本")
            
            # 打乱最终的数据列表
            random.shuffle(self.data_list)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.data_list, f)
            print(f"数据已保存: 总共 {len(self.data_list)} 个样本")

        self.output_path = './data/robustness/temporal_consistency/temporal_videos_new/'

        self.new_data_list = []

        errors_videos = ['OKHVL', "ZZ89F", 'SUG5S', 'J95U1', 'D1WYU', 'B0XI9', 'TKAUR']
        for idx in range(len(self.data_list)):
            video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
            print('video_path:', video_path)
            video_id = video_path.split('/')[-1].split('.')[0]
            if video_id in errors_videos:
                continue
               
            video_drop_path = os.path.join(self.output_path, video_id + '_drop.mp4')
            video_shuffle_path = os.path.join(self.output_path, video_id + '_shuffle.mp4')

            if not os.path.exists(video_drop_path):
                self.process_video_drop(video_path, output_path_drop=video_drop_path, drop_ratio=0.2)
            

            self.new_data_list.append({'origin_video_path':video_path,
                                        'video_path': video_path, 
                                    'data':self.data_list[idx]['data'],
                                    'task_type':self.data_list[idx]['task_type']})

            if not os.path.exists(video_shuffle_path):
                self.process_video_shuffle(video_path, output_path_shuffle=video_shuffle_path, drop_ratio=0.2)
            self.new_data_list.append({'origin_video_path':video_path,
                                        'video_path': video_shuffle_path, 
                                    'data':self.data_list[idx]['data'],
                                    'task_type': self.data_list[idx]['task_type']})

        
    def __len__(self):
        return len(self.new_data_list)
    
    def process_video_drop(self, video_path, output_path_drop, drop_ratio=0.2):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("错误：无法打开视频文件")
            return
        
        # 获取视频属性
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
       
        total_frames = len(frames)
        print(f"原始帧数: {total_frames}")
        
        # 随机丢弃帧
        drop_num = int(total_frames * drop_ratio) 
        keep_indices = sorted(random.sample(range(total_frames), total_frames - drop_num))  # 保留的帧索引
        dropped_frames = [frames[i] for i in keep_indices]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out_drop = cv2.VideoWriter(output_path_drop, fourcc, fps, (width, height))
        for frame in dropped_frames:
            out_drop.write(frame)
        out_drop.release()
        print(f"丢弃帧后视频已保存到: {output_path_drop}, 帧数: {len(dropped_frames)}")
        

    def process_video_shuffle(self, video_path, output_path_shuffle, drop_ratio=0.2):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("错误：无法打开视频文件")
            return
        
        # 获取视频属性
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
       
        total_frames = len(frames)
        print(f"原始帧数: {total_frames}")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
       
        # 打乱帧顺序
        shuffled_frames = frames.copy()
        random.shuffle(shuffled_frames) 
        
        # 保存打乱帧顺序后的视频
        out_shuffle = cv2.VideoWriter(output_path_shuffle, fourcc, fps, (width, height))
        for frame in shuffled_frames:
            out_shuffle.write(frame)
        out_shuffle.release()
        print(f"打乱帧顺序后视频已保存到: {output_path_shuffle}, 帧数: {len(shuffled_frames)}")


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
        
        question, answer = self.qa_template(self.new_data_list[idx]['data'])

        if self.dataset_id == "TemporalConsistencyDataset":
            video_path = self.new_data_list[idx]['video_path']
        else:
            video_path = self.new_data_list[idx]['origin_video_path']

        if self.method_hook:
            return self.method_hook.run(VideoTxtSample( 
            video_frames=None,
            video_path=video_path,
            question=question, 
            answer=answer,
            extra= self.new_data_list[idx]['task_type']
            ))
            
        return VideoTxtSample( 
            video_frames=None,
            video_path=video_path,
            question=question, 
            answer=answer,
            extra=self.new_data_list[idx]['task_type']
        )
