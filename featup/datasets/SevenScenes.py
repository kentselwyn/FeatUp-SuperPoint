import os
from os.path import join, exists
from PIL import Image
from torch.utils.data import Dataset


class SevenScenes(Dataset):
    """
    7Scenes Dataset Loader
    
    Dataset structure:
    7scenes/[scene_name]/[sequence_name]/frame-[frame_number].color.png
    
    Split files:
    7scenes/[scene_name]/TestSplit.txt
    7scenes/[scene_name]/TrainSplit.txt
    """
    
    SCENE_NAMES = [
        'chess',
        'fire', 
        'heads',
        'office',
        'pumpkin',
        'redkitchen',
        'stairs'
    ]
    
    def __init__(self, root, scene_name, split, transform, target_transform=None, include_labels=False):
        """
        Initialize 7Scenes dataset
        
        Args:
            root (str): Root directory containing 7scenes folder
            scene_name (str): Scene name (e.g., 'chess', 'fire', etc.)
            split (str): 'train' or 'test'
            transform: Image transform function
            target_transform: Target transform function (optional)
            include_labels (bool): Whether to include labels (not used for 7Scenes)
        """
        super(SevenScenes, self).__init__()
        
        self.root = root
        self.scene_name = scene_name
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.include_labels = include_labels
        
        # Validate scene name
        if scene_name not in self.SCENE_NAMES:
            raise ValueError(f"Invalid scene name '{scene_name}'. Must be one of {self.SCENE_NAMES}")
            
        # Validate split
        if split not in ['train', 'test']:
            raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'test'")
        
        # Set up paths
        self.scene_dir = join(root, '7scenes', scene_name)
        if not exists(self.scene_dir):
            raise FileNotFoundError(f"Scene directory not found: {self.scene_dir}")
        
        # Load split file
        split_file = join(self.scene_dir, f"{'Train' if split == 'train' else 'Test'}Split.txt")
        if not exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        # Parse split file and collect image paths
        self.image_paths = []
        self._load_split_file(split_file)
        
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found for scene '{scene_name}', split '{split}'")
    
    def _load_split_file(self, split_file):
        """Load sequence names from split file and collect all image paths"""
        with open(split_file, 'r') as f:
            sequences = [line.strip() for line in f.readlines() if line.strip()]
            sequences = [f"seq-{int(line[-1]):02d}" for line in sequences]
        
        for sequence in sequences:
            sequence_dir = join(self.scene_dir, sequence)
            if not exists(sequence_dir):
                print(f"Warning: Sequence directory not found: {sequence_dir}")
                continue
            
            # Find all color images in this sequence
            color_images = []
            for filename in os.listdir(sequence_dir):
                if filename.endswith('.color.png'):
                    color_images.append(filename)
            
            # Sort by frame number
            color_images.sort(key=lambda x: int(x.split('-')[1].split('.')[0]))
            
            # Add full paths
            for img_file in color_images:
                img_path = join(sequence_dir, img_file)
                self.image_paths.append(img_path)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get item from dataset
        
        Returns:
            dict: Dictionary containing:
                - 'img': Transformed image tensor
                - 'img_path': Path to the image file
        """
        if idx >= len(self.image_paths):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.image_paths)}")
        
        image_path = self.image_paths[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        batch = {
            "img": image,
            "img_path": image_path
        }
        
        return batch
    
    def get_scene_info(self):
        """Get information about the loaded scene"""
        return {
            'scene_name': self.scene_name,
            'split': self.split,
            'num_images': len(self.image_paths),
            'scene_dir': self.scene_dir
        }
    
    @classmethod
    def get_available_scenes(cls, root):
        """Get list of available scenes in the dataset root"""
        scenes_root = join(root, '7scenes')
        if not exists(scenes_root):
            return []
        
        available_scenes = []
        for scene_name in cls.SCENE_NAMES:
            scene_dir = join(scenes_root, scene_name)
            if exists(scene_dir):
                available_scenes.append(scene_name)
        
        return available_scenes