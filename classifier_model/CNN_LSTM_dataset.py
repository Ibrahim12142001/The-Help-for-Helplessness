import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import random

class HelplessnessVideoDataset(Dataset):
    """
    Reads directory structure:
        root_dir/
            extreme-helpless/
                clipA/
                    frame_000.jpg
                    ...
            little_helplessness/
                ...
            no-helpless/
                ...
    Returns folder paths for each video clip.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.video_folders = []

        categories = ['extreme-helpless', 'little_helplessness', 'no-helpless']
        for category in categories:
            category_dir = os.path.join(root_dir, category)
            if not os.path.isdir(category_dir):
                continue

            for video_folder in sorted(os.listdir(category_dir)):
                video_path = os.path.join(category_dir, video_folder)
                if os.path.isdir(video_path):
                    self.video_folders.append(video_path)

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, index):
        return self.video_folders[index]


class TransformableSequenceSubset(Dataset):
    """
    Loads frames, converts to grayscale (1-channel),
    applies the same random transform to every frame in that video,
    stacks frames into a tensor of shape (T, 1, H, W),
    and returns (sequence, label).
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        video_path = self.subset[index]
        frame_files = sorted(os.listdir(video_path))
        sequence = []

        # We need to ensure the same random transform is applied to all frames in the clip.
        seed = random.randint(0, 99999)
        for frame_name in frame_files:
            frame_path = os.path.join(video_path, frame_name)
            # Convert the images to grayscale because the 2D seems to work better with it
            frame = Image.open(frame_path).convert('L')
            if self.transform:
                random.seed(seed)
                torch.manual_seed(seed)
                frame = self.transform(frame)
            sequence.append(frame)

        # Stack frames: resulting shape (T, 1, H, W)
        sequence = torch.stack(sequence, dim=0)

        # Get the label from folder names 
        path_parts = video_path.split(os.sep)
        category = path_parts[-2]
        label_map = {
            'no-helpless': 0,
            'little_helplessness': 1,
            'extreme-helpless': 2
        }
        label = label_map.get(category, -1)
        return sequence, label

