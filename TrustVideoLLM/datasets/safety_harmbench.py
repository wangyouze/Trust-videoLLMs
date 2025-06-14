
import os
from typing import Optional, Sequence
from .base import BaseDataset
import pandas as pd
import json
from TrustVideoLLM.utils.registry import registry
from TrustVideoLLM.methods.base import BaseMethod
from TrustVideoLLM import VideoTxtSample



@registry.register_dataset()
class HarmBenchDataset(BaseDataset):
    dataset_ids: Sequence[str] = [
        "HarmBenchDataset",
    ]
  
    def __init__(self,  dataset_id, method_hook: Optional[BaseMethod] = None, data_dir=None, video_dir=None):
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)

        if not os.path.exists('./data/safety/Harmbench/sampled_harmbench.json'):
            df = pd.read_csv(data_dir)
            sampled_data = self.stratified_sample(df, target_size=100, category_column="SemanticCategory")
            self.data_list = sampled_data["Behavior"].tolist()
            with open('./data/safety/Harmbench/sampled_harmbench.json', 'w', encoding='utf-8') as f:
                json.dump(self.data_list, f)
        else:
            with open('./data/safety/Harmbench/sampled_harmbench.json', 'r', encoding='utf-8') as f:
                self.data_list = json.load(f)

        self.video_dir = video_dir

    
    def __len__(self):
        return len(self.data_list)
    

    def stratified_sample(self, df, target_size, category_column):
        # Get the unique categories and their counts
        category_counts = df[category_column].value_counts()
        categories = category_counts.index
        num_categories = len(categories)
        
        # Target number of samples per category (as even as possible)
        target_per_category = target_size // num_categories
        remaining_samples = target_size % num_categories
        
        sampled_dfs = []
        for category in categories:
            category_df = df[df[category_column] == category]
            category_size = len(category_df)
            
            # If the category has fewer rows than the target, take all rows
            if category_size <= target_per_category:
                sampled_dfs.append(category_df)
            else:
                # Sample the target number of rows (add 1 if there are remaining samples)
                sample_size = target_per_category + (1 if remaining_samples > 0 else 0)
                if remaining_samples > 0:
                    remaining_samples -= 1
                sampled_dfs.append(category_df.sample(n=sample_size, random_state=42))
        
        # Concatenate all sampled dataframes
        result = pd.concat(sampled_dfs, ignore_index=True)
        
        # If we have more than target_size due to rounding, trim the excess
        if len(result) > target_size:
            result = result.sample(n=target_size, random_state=42)
        
        # If we have fewer than target_size, sample additional rows randomly
        if len(result) < target_size:
            remaining = target_size - len(result)
            remaining_df = df[~df.index.isin(result.index)]
            additional_samples = remaining_df.sample(n=remaining, random_state=42)
            result = pd.concat([result, additional_samples], ignore_index=True)
        
        return result


    def __getitem__(self, idx):
        
        video_path = self.video_dir
        print('video_path:', video_path)
        
       
        question = self.data_list[idx]
        print('prompt=', question)
        answer = None

        if self.method_hook:
            return self.method_hook.run(VideoTxtSample( 
            video_frames=None,
            video_path=video_path,
            question=question, 
            answer=answer,
            task_type= ''
            ))
            
        return VideoTxtSample( 
            video_frames=None,
            video_path=video_path,
            question=question, 
            answer=answer,
            task_type=''
        )