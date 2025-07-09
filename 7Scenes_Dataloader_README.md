# 7Scenes Dataset Loader for FeatUp

This document describes the 7Scenes dataset loader implementation for the FeatUp framework.

## Overview

The 7Scenes dataset loader (`SevenScenes`) provides seamless integration with the FeatUp training pipeline, supporting all 7 scenes from the Microsoft 7Scenes dataset.

## Dataset Structure

The loader expects the following directory structure:

```
data_root/
├── 7_scenes/
│   ├── chess/
│   │   ├── TrainSplit.txt
│   │   ├── TestSplit.txt
│   │   ├── seq-01/
│   │   │   ├── frame-000000.color.png
│   │   │   ├── frame-000001.color.png
│   │   │   └── ...
│   │   ├── seq-02/
│   │   └── ...
│   ├── fire/
│   ├── heads/
│   ├── office/
│   ├── pumpkin/
│   ├── redkitchen/
│   └── stairs/
```

## Supported Scenes

- `chess`
- `fire`
- `heads`
- `office`
- `pumpkin`
- `redkitchen`
- `stairs`

## Usage

### Direct Usage

```python
from featup.datasets.SevenScenes import SevenScenes
import torchvision.transforms as T

transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224), 
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load chess scene training data
dataset = SevenScenes(
    root='/path/to/data',  # Should contain 7_scenes/ folder
    scene_name='chess',
    split='train',         # 'train' or 'test'
    transform=transform
)
```

### Via FeatUp get_dataset Function

```python
from featup.datasets.util import get_dataset

dataset = get_dataset(
    dataroot='/path/to/data',
    name='7scenes_chess',  # Format: 7scenes_[scene_name]
    split='train',         # 'train' or 'test'
    transform=transform,
    target_transform=None,
    include_labels=False
)
```

### Configuration File Usage

Use the provided example configuration:

```yaml
# featup/configs/7scenes_example.yaml
dataset: "7scenes_chess"  # or any other scene
split: "train"           # or "test"
pytorch_data_dir: "/path/to/your/data"
```

Run training with:
```bash
python train_implicit_upsampler.py --config-name=7scenes_example
```

Or override specific parameters:
```bash
python train_implicit_upsampler.py dataset=7scenes_fire split=test
```

## Dataset Features

✓ **All 7 Scenes Supported**: chess, fire, heads, office, pumpkin, redkitchen, stairs  
✓ **Train/Test Splits**: Automatically loads from TrainSplit.txt and TestSplit.txt  
✓ **Automatic Frame Discovery**: Finds and sorts all frame-*.color.png files  
✓ **Robust Error Handling**: Graceful handling of missing files/directories  
✓ **FeatUp Compatible**: Works seamlessly with existing FeatUp pipeline  
✓ **Memory Efficient**: Loads images on-demand  
✓ **Standard Output Format**: Returns `{'img': tensor, 'img_path': str}`  

## API Reference

### SevenScenes Class

```python
class SevenScenes(Dataset):
    def __init__(self, root, scene_name, split, transform, 
                 target_transform=None, include_labels=False):
        """
        Args:
            root (str): Root directory containing 7_scenes folder
            scene_name (str): Scene name ('chess', 'fire', etc.)
            split (str): 'train' or 'test'
            transform: Image transform function
            target_transform: Target transform (optional)
            include_labels (bool): Include labels (not used for 7Scenes)
        """
```

### Utility Methods

```python
# Get dataset information
info = dataset.get_scene_info()
# Returns: {'scene_name': str, 'split': str, 'num_images': int, 'scene_dir': str}

# Get available scenes in dataset
available = SevenScenes.get_available_scenes('/path/to/data')
# Returns: List of available scene names
```

## Error Handling

The loader includes comprehensive error handling:

- **Invalid scene names**: Raises `ValueError` with list of valid scenes
- **Invalid splits**: Raises `ValueError` for splits other than 'train'/'test'  
- **Missing directories**: Raises `FileNotFoundError` with specific path
- **Missing split files**: Raises `FileNotFoundError` for TrainSplit.txt/TestSplit.txt
- **No images found**: Raises `RuntimeError` if no valid images in split
- **Image loading errors**: Raises `RuntimeError` with specific image path

## Integration with FeatUp Training

The 7Scenes loader integrates seamlessly with FeatUp's training pipeline:

1. **SuperPoint + 7Scenes**: Combine with SuperPoint featurizer for keypoint-aware feature upsampling
2. **Multiple Scenes**: Train on different scenes by changing the dataset name
3. **Custom Preprocessing**: Apply scene-specific transforms if needed
4. **Evaluation**: Use test split for evaluation after training

## Example Training Commands

```bash
# Train SuperPoint on chess scene
python train_implicit_upsampler.py dataset=7scenes_chess model_type=superpoint

# Train DINO on office scene  
python train_implicit_upsampler.py dataset=7scenes_office model_type=dino16

# Use test split for evaluation
python train_implicit_upsampler.py dataset=7scenes_fire split=test steps=100
```

## Files Created/Modified

1. **`featup/datasets/SevenScenes.py`**: Main dataset implementation
2. **`featup/datasets/util.py`**: Updated to include 7Scenes support
3. **`featup/configs/7scenes_example.yaml`**: Example configuration
4. **`test_7scenes_dataloader.py`**: Test and demonstration script

This implementation provides a robust, feature-complete dataloader for the 7Scenes dataset that integrates seamlessly with the FeatUp framework.
