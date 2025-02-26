# Acute Traumatic Physical Injury Segmentation

  A convolutional neural network designed to segment acute traumatic 
  physical injuries (such as lacerations and stab wounds) from medical 
  images with high precision. This U-Net based architecture implements 
  custom loss functions and weighting strategies to optimize performance 
  on limited training data.

# Overview

  This repository contains a complete solution for medical wound segmentation 
  using deep learning. The model is trained to identify the precise boundaries 
  of open wounds in medical imagery, which has applications in wound assessment, 
  measurement, and treatment planning. Specifically, its intended use is for 
  automated emmergency injury treatment via computer vision embedded robotics. 

### Key Features

- **U-Net Architecture** optimized for wound segmentation
- **Custom Loss Functions** combining Dice coefficient and IoU with class weighting
- **Data Augmentation Pipeline** to enhance limited training data
- **Medical-specific Preprocessing** techniques

## Model Architecture

The implementation uses a modified U-Net architecture with:

- Encoder-decoder structure for preserving spatial information
- Dropout and BatchNormalization layers to prevent overfitting
- Skip connections to maintain fine detail in segmentation
- Sigmoid activation for binary segmentation output

## Dataset

- **Custom Dataset**: Manually collected and annotated by the author
- **Initial Size**: 40 unique wound samples
- **Annotations**: Pixel-level masks of wound boundaries
- **Data Challenges**: Limited sample size addressed through extensive augmentation

## Data Augmentation Techniques

To overcome the limited training data, extensive offline augmentation was implemented:

- **Geometric Transformations**: Rotation, scaling, flipping, cropping, shearing
- **Appearance Modifications**: Contrast adjustment, color shifts, brightness variation
- **Advanced Techniques**: 
  - Superimposition
  - Noise injection
  - Saturation adjustments
  - Region of interest removal
  - Background variation

These augmentation strategies expanded the effective training set while teaching the model to be invariant to irrelevant variations.

## Loss Functions and Training

Custom loss functions were designed to address the specific challenges of wound segmentation:

- **Weighted Dice Coefficient Loss**: Addresses class imbalance between wound and background pixels
- **IoU Loss**: Focuses on the overlap between predicted and ground truth segmentation
- **Combined Loss**: Weighted combination for optimal training signal

```python
def combined_dice_iou_loss(y_true, y_pred, smooth=1e-6, dice_weight=0.5, 
                          iou_weight=0.5, wound_weight=0.75):
    unweighted_dice_loss = dice_coef_loss(y_true, y_pred, smooth)
    weighted_dice_loss = weighted_dice_coef_loss(y_true, y_pred, smooth)
    combined_dice_loss = wound_weight * weighted_dice_loss + (1 - wound_weight) * unweighted_dice_loss

    iou_loss_val = iou_loss(y_true, y_pred, smooth)

    combined_loss = dice_weight * combined_dice_loss + iou_weight * iou_loss_val
    return combined_loss
```

## Performance

The model was trained with early stopping based on validation performance to prevent overfitting. Performance metrics include:

- Dice Coefficient
- IoU (Intersection over Union)
- Accuracy
- Custom weighted metrics

## Usage

### Installation

```bash
git clone https://github.com/yourusername/wound-segmentation.git
cd wound-segmentation
pip install -r requirements.txt
```

### Training

```bash
python train.py --train_dir path/to/training --val_dir path/to/validation
```

### Inference

```python
from keras.models import load_model
import numpy as np
from PIL import Image
from keras.utils import custom_object_scope

# Load custom functions
with custom_object_scope({
    'weighted_dice_coef_loss': weighted_dice_coef_loss,
    'dice_coef': dice_coef,
    'iou': iou,
    'combined_dice_iou_loss': combined_dice_iou_loss
}):
    model = load_model('path/to/model.keras')

# Preprocess image
img = Image.open('wound_image.jpg')
img = img.resize((256, 256))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
mask = model.predict(img_array)
```

## Future Improvements

- Expansion of the training dataset with more diverse wound types
- Implementation of transfer learning from larger medical imaging datasets
- Exploration of attention mechanisms to improve boundary precision
- Real-time segmentation optimization

## Some Testing Samples

![june05v1test](https://github.com/user-attachments/assets/fe1bd2c5-72c1-42ed-b381-29c0741a1431)
![testdice7398](https://github.com/user-attachments/assets/484480c7-5da7-4531-831c-e461460d4046)
![june11v7](https://github.com/user-attachments/assets/dfd717c5-8f41-4f6f-8024-e9c3fc1640f3)
![livefeedtest1](https://github.com/user-attachments/assets/a9570e1f-03d4-4747-b6bc-bbc5570c538e)
![oldmodel](https://github.com/user-attachments/assets/24f60e4e-7112-4142-b3aa-0cac6ae520e0)

## License

[MIT License]

## Contact

[Dylan Somra]
dylan.csomra@gmail.com 
https://www.linkedin.com/in/dylan-somra-a425391a0/]
