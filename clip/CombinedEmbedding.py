
import torch
import clip
from PIL import Image

# Load the pre-trained CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Define your text and image
text = "This is the image of a toilet"
image_path = "sochi_fake_20.jpg"

# Preprocess the text and image
text_tokens = clip.tokenize([text]).to(device)
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

# Generate embeddings
with torch.no_grad():
    text_embeddings = model.encode_text(text_tokens)
    image_embeddings = model.encode_image(image)

# Normalize the embeddings
text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)

# Option 1: Concatenate embeddings
combined_embeddings = torch.cat((text_embeddings, image_embeddings), dim=1)

# Option 2: Average embeddings
# combined_embeddings = (text_embeddings + image_embeddings) / 2

print("Combined Embeddings:", combined_embeddings)