import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class HelplessnessVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.video_folders = []

        # Read all video sequences in the extracted frames folders
        categories = ['extreme-helpless', 'little_helplessness', 'no-helpless']
        for category in categories:
            category_dir = os.path.join(root_dir, category)
            if not os.path.exists(category_dir):
                print(f"Warning: Category folder {category_dir} does not exist.")
                continue
            for video_folder in sorted(os.listdir(category_dir)):
                video_path = os.path.join(category_dir, video_folder)
                if os.path.isdir(video_path):
                    self.video_folders.append(video_path)

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, index):
        video_path = self.video_folders[index]

        # Each video is a sequence of frames, so need to get each frame
        frame_files = sorted(os.listdir(video_path))
        sequence = []
        random_state = torch.get_rng_state()
        for frame_name in frame_files:
            frame_path = os.path.join(video_path, frame_name)
            frame = Image.open(frame_path)
            if self.transform:
                # To allow augmentation, we need to apply the same "random" transformation to each frame
                torch.set_rng_state(random_state)
                frame = self.transform(frame)
            sequence.append(frame)

        # Convert the sequence from list of tensors to (sequence_length, channels, height, width) tensor
        sequence = torch.stack(sequence)
        sequence = torch.transpose(sequence, 0, 1)  # REMOVE THIS IF YOU NEED THE SEQUENCE_LENGTH AND CHANNELS DIMENSIONS SWITCHED!

        # Retrieve the level of helplessness label from path of video
        split_path = video_path.split(os.sep)  # Changed '/' to os.sep for cross-platform compatibility
        print(f"split_path: {split_path}")  # Add this for debugging

        # Ensure that split_path has at least 2 components
        if len(split_path) >= 2:
            category = split_path[-2]  # category folder is second last in path
        else:
            raise ValueError(f"Path structure issue: {video_path}")

        label = -1
        if category == 'no-helpless':
            label = 0
        elif category == 'little_helplessness':
            label = 1
        elif category == 'extreme-helpless':
            label = 2

        return sequence, label


# this brilliant idea comes from: https://stackoverflow.com/questions/51782021/how-to-use-different-data-augmentation-for-subsets-in-pytorch
class TransformableSequenceSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        video_path = self.subset[index]

        # Each video is a sequence of frames, so need to get each frame
        frame_files = sorted(os.listdir(video_path))
        sequence = []
        random_state = torch.get_rng_state()
        for frame_name in frame_files:
            frame_path = os.path.join(video_path, frame_name)
            frame = Image.open(frame_path)
            if self.transform:
                # To allow augmentation, we need to apply the same "random" transformation to each frame
                torch.set_rng_state(random_state)
                frame = self.transform(frame)
            sequence.append(frame)

        # Convert the sequence from list of tensors to (sequence_length, channels, height, width) tensor
        sequence = torch.stack(sequence)
        sequence = torch.transpose(sequence, 0,
                                   1)  # REMOVE THIS IF YOU NEED THE SEQUENCE_LENGTH AND CHANNELS DIMENSIONS SWITCHED!
        # Need to transpose the sequence_length and channels for our model to use 3D Convolutions

        # Retrieve the level of helplessness label from path of video
        split_path = video_path.split('/')
        category = split_path[-2]  # category folder is second last in path
        label = -1
        if category == 'no-helpless':
            label = 0
        elif category == 'little_helplessness':
            label = 1
        elif category == 'extreme-helpless':
            label = 2

        return sequence, label