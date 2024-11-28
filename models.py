import tensorflow as tf
from tensorflow.keras.models import load_model
import torch
from torchvision import transforms
from model_architecture import FruitClassifierCNN  # Assuming the PyTorch architecture is in `model_architecture.py`

# Load TensorFlow model
flower_model = load_model("./flower_classification_model.h5")

# Load PyTorch model
num_classes = 141
fruit_model = FruitClassifierCNN(num_classes)
fruit_model.load_state_dict(torch.load("./fruit_classifier_model.pth", map_location=torch.device('cpu')))
fruit_model.eval()

# Define PyTorch transforms
fruit_transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])
