import torch
import torch.nn.functional as F
import clip
from PIL import Image
import torchmetrics

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

print(type(model))
print(type(preprocess))

# switch between 0 and 416
image = preprocess(Image.open("data/images/416.jpg")).unsqueeze(0).to(device)
text = clip.tokenize([
                        "The person has high cheekbones, and pointy nose. She is wearing lipstick.",
                        "This man is attractive and has bushy eyebrows, and black hair.",
                        "The man has black hair. He is attractive. He is the Witcher.",
                        "He is Henry Cavill."
                    ]).to(device)

print(clip.available_models())

with torch.no_grad():
    image_features = model.encode_image(image)

    print(image_features.expand(16, -1).shape)
    text_features = model.encode_text(text)
    print(text_features.shape)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
print(f"Similarity:", (100.0 * text_features @ text_features.T))
print(f"Cosine sims:", F.cosine_similarity(text_features, text_features))
print(f"Eucledian sims:", torchmetrics.functional.pairwise_euclidean_distance(image_features, text_features))