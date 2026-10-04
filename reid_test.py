from PIL import Image
import torch
from torchvision import transforms
import torchreid
import torch.nn.functional as F


# -----------------------------
# Load OSNet
# -----------------------------

model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=1000,
    pretrained=True
)

model.eval()


# -----------------------------
# Prepare images
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def get_embedding(image_path):

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        embedding = model(image_tensor)

    if isinstance(embedding, tuple):
        embedding = embedding[0]

    return embedding


# -----------------------------
# Generate embeddings
# -----------------------------

person_a = get_embedding("person_a.jpg")
person_b = get_embedding("person_b.jpg")
person_c = get_embedding("person_c.jpg")


# -----------------------------
# Compare embeddings
# -----------------------------

similarity_ab = F.cosine_similarity(person_a, person_b)
similarity_ac = F.cosine_similarity(person_a, person_c)


print("\n--- ReID Results ---")

print(f"You vs You:     {similarity_ab.item():.4f}")
print(f"You vs Brother: {similarity_ac.item():.4f}")