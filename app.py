import streamlit as st
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import gdown

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet-50 expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

# Download the model file from Google Drive
file_id = '1T-tIt5VUWlSuF16gOAAnXLD7Rkz7WWaN'  # Extracted from the Google Drive link
url = f'https://drive.google.com/uc?id={file_id}'
output = 'ai_vs_real_model.pth'
gdown.download(url, output, quiet=False)

# Assuming model is defined or loaded elsewhere in your script
from torchvision import models

# Initialize the model
model = models.resnet50()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Assuming the output size in the checkpoint is 2
# Load the state dict into the model
model.load_state_dict(torch.load(output, map_location=torch.device('cpu')))

model.eval()

def predict(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1)
    return 'AI-generated' if predicted_class == 0 else 'Real'

# Streamlit interface
st.title('AI and Real Image Detector')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    if st.button('Predict'):
        result = predict(image)
        st.write(f"The image is classified as: **{result}**")
