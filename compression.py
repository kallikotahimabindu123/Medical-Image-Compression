import torch
import os
from model import AutoEncoder, LegacyAutoEncoder
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

IMG_SIZE = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "autoencoder.pth")

checkpoint = torch.load(MODEL_PATH, map_location=device)

model = AutoEncoder().to(device)

try:
    model.load_state_dict(checkpoint)
except RuntimeError:
    if any(key.startswith("encoder.") for key in checkpoint.keys()):
        model = LegacyAutoEncoder().to(device)
        model.load_state_dict(checkpoint)
        print("Loaded legacy checkpoint format. Retrain with current model for best quality.")
    else:
        raise

model.eval()

transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE,IMG_SIZE), interpolation=InterpolationMode.LANCZOS, antialias=True),
    transforms.ToTensor()
])

def compress_image(path):

    img=Image.open(path)
    img_t=transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output=model(img_t).cpu()

    return output