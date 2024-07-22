# import torch
# import clip
# from PIL import Image

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

# image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
# text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
    
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Labes: a diagram, a dog, a cat ", )
# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]





import streamlit as st
from PIL import Image
import torch
import clip

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

st.title("CLIP Image and Text Analysis")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Text inputs
    text1 = st.text_input("Enter first text", " ")
    text2 = st.text_input("Enter second text", " ")
    text3 = st.text_input("Enter third text", " ")
    
    if st.button('Analyze'):
        # Preprocess image and texts
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_inputs = clip.tokenize([text1, text2, text3]).to(device)
        
        with torch.no_grad():
            # Encode image and texts
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
            
            # Calculate logits and probabilities
            logits_per_image, logits_per_text = model(image_input, text_inputs)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # Display probabilities
        st.write("Labels:", [text1, text2, text3])
        st.write("Label probs:", probs[0])
