import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=2
)

model.load_state_dict(torch.load("vit_deepfake.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

st.title("Deepfake Detection using Vision Transformer")

uploaded = st.file_uploader("Upload an image")

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image)

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img).logits
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    label = "FAKE" if pred.item()==1 else "REAL"

    st.write(f"Prediction: {label}")
    st.write(f"Confidence: {confidence.item()*100:.2f}%")