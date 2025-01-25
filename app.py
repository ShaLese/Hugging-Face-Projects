import sys
print("Python version:", sys.version)
print("Python path:", sys.path)

try:
    import torch
    print("PyTorch version:", torch.__version__)
except ImportError as e:
    print("Error importing torch:", str(e))
    print("Detailed error information:", sys.exc_info())

import streamlit as st
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import time
import numpy as np

# CIFAR-10 classes
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

# Define the ResNet architecture (same as in the notebook)
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CustomResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self.make_layer(64, 2, stride=1)
        self.layer2 = self.make_layer(128, 2, stride=2)
        self.layer3 = self.make_layer(256, 2, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Load the trained model
@st.cache_resource
def load_model():
    model = CustomResNet()
    model.load_state_dict(torch.load('resnet_cifar10.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Image preprocessing
def preprocess_image(image):
    # Convert grayscale to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    return transform(image).unsqueeze(0)

# Streamlit UI
def main():
    st.title('Image Classification with ResNet')
    st.write('Upload an image or take a photo to classify it into one of these categories:', CLASSES)
    
    # Load model
    try:
        model = load_model()
        st.success('Model loaded successfully!')
    except Exception as e:
        st.error(f'Error loading model: {str(e)}')
        return
    
    # Add a radio button for input selection
    input_method = st.radio(
        "Choose input method:",
        ("Upload Image", "Take Photo")
    )
    
    image = None
    
    if input_method == "Upload Image":
        # File uploader
        uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
    else:
        # Camera input
        camera_image = st.camera_input("Take a photo")
        if camera_image is not None:
            image = Image.open(camera_image)
    
    if image is not None:
        try:
            # Display image
            st.image(image, caption='Input Image', use_container_width=True)
            
            # Make prediction
            start_time = time.time()
            
            # Preprocess image
            processed_image = preprocess_image(image)
            
            # Get prediction
            with torch.no_grad():
                outputs = model(processed_image)
                probabilities = F.softmax(outputs, dim=1)[0]
                prediction_time = time.time() - start_time
                
                # Get top 3 predictions
                top3_prob, top3_indices = torch.topk(probabilities, 3)
                
                # Create columns for predictions
                st.write('### Predictions')
                cols = st.columns(3)
                
                # Display results with progress bars
                for i, (prob, idx) in enumerate(zip(top3_prob, top3_indices)):
                    with cols[i]:
                        st.metric(
                            label=f"{i+1}. {CLASSES[idx]}",
                            value=f"{prob.item()*100:.1f}%"
                        )
                        st.progress(prob.item())
                
                st.write(f'Prediction time: {prediction_time:.3f} seconds')
                
        except Exception as e:
            st.error(f'Error processing image: {str(e)}')

if __name__ == '__main__':
    main()
