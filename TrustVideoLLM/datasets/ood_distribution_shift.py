import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2

class CapERADataset(Dataset):
    def __init__(self, root_dir, labels_file, transform=None):
        """
        初始化 CapERA 数据集
        Args:
            root_dir (str): 视频文件所在目录
            labels_file (str): 标签文件路径
            transform (callable, optional): 视频帧的预处理方法
        """
        self.root_dir = root_dir
        self.labels = self._load_labels(labels_file)
        self.transform = transform

    def _load_labels(self, labels_file):
        """
        加载标签文件
        """
        labels = {}
        with open(labels_file, 'r') as f:
            for line in f:
                video_name, label = line.strip().split(',')
                labels[video_name] = label
        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        获取单个样本
        """
        video_name = list(self.labels.keys())[idx]
        label = self.labels[video_name]
        video_path = os.path.join(self.root_dir, video_name)

        # 加载视频帧
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()

        frames = torch.stack(frames)
        label = torch.tensor(int(label))  # 将情感标签转换为数值

        return {'frames': frames, 'label': label}



