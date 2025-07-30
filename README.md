# 🐾 Nose Segmentation in Animal Faces using DeepLabV3

This project focuses on semantic segmentation of the **nose** in animal (e.g. cat, dog) images using the **DeepLabV3** architecture with a **ResNet-50** backbone in PyTorch.

## 📌 Project Overview

- 🔍 Objective: Detect and segment the nose region in animal faces.
- 🧠 Model: DeepLabV3 with `num_classes=2` (background & nose)
- 🏷 Dataset: Custom-annotated dataset in COCO format with bounding boxes converted to binary masks.
- 🎯 Application: Can be extended for animal identification, health tracking, and facial keypoint detection.

## 📁 Project Structure

```
petnerub/
│
├── annotatedimages/
│   ├── Images/              # Raw animal images
│   ├── annotations/         # COCO-style JSON annotations (e.g. instances_default.json)
│   └── masks/               # Auto-generated nose masks (.png)
│
├── models/
│   └── deeplabv3_nose.pth   # Trained model weights (state_dict)
│
├── datasets/
│   └── Testdata/            # Test images for inference
│
├── training_script.ipynb    # Training notebook (Colab-friendly)
└── inference_script.ipynb   # Notebook for inference and visualization
```

## 🏗 How to Train

1. Generate binary masks from COCO annotations using the provided script.
2. Use the `NoseSegmentationDataset` class to load image/mask pairs.
3. Train the model using DeepLabV3:
```python
torch.save(model.state_dict(), "models/deeplabv3_nose.pth")
```

## 🔍 How to Predict

Load the model:
```python
import torchvision.models.segmentation as models
model = models.deeplabv3_resnet50(pretrained=False, num_classes=2)
model.load_state_dict(torch.load("models/deeplabv3_nose.pth", map_location="cpu"))
model.eval()
```

Predict on a new image and overlay the mask using Matplotlib.

## 📊 Evaluation

Model performance can be evaluated using pixel accuracy or IoU between predicted and ground truth masks. A basic accuracy script is included.

## 📦 Dependencies

- Python 3.9+
- PyTorch
- torchvision
- NumPy
- Pillow
- Matplotlib
- OpenCV

## ✍️ Author

**[Mahya Karamad Kasmei]**  
M.Eng. Systems & Technology — McMaster University  
Email: mahya.karamad@gmail.com

## 📄 License

This project is licensed under the **Apache 2.0 License** – see the `LICENSE` file for details.
