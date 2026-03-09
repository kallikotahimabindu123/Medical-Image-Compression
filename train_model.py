import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from model import AutoEncoder

IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 40
LEARNING_RATE = 1e-3

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE,IMG_SIZE), interpolation=InterpolationMode.LANCZOS, antialias=True),
    transforms.ToTensor()
])

dataset = ImageFolder("dataset",transform=transform)
loader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "autoencoder.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder().to(device)
mse_loss = torch.nn.MSELoss()
l1_loss = torch.nn.L1Loss()
optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

best_loss = float("inf")

for epoch in range(EPOCHS):

    loss_total=0

    for img,_ in loader:
        img = img.to(device)

        output=model(img)
        loss=0.7*mse_loss(output,img)+0.3*l1_loss(output,img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total+=loss.item()

    avg_loss = loss_total / max(len(loader), 1)
    scheduler.step(avg_loss)

    if avg_loss < best_loss:
        best_loss = avg_loss
        os.makedirs(os.path.dirname(MODEL_PATH),exist_ok=True)
        torch.save(model.state_dict(),MODEL_PATH)

    print("Epoch:",epoch+1,"AvgLoss:",round(avg_loss,6),"Best:",round(best_loss,6))

print("Training complete. Best model saved to", MODEL_PATH)