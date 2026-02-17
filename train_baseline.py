import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os

# --- CONFIGURATION ---
# Auto-detect GPU (RTX 3050)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5           # Enough for demo purposes (~80% acc)
BATCH_SIZE = 32
LR = 0.001           # Learning Rate

def train_and_save():
    print(f"[Setup] Training on {DEVICE}. This will take ~2-5 minutes...")
    
    # 1. SETUP MODEL
    # Load pre-trained ImageNet weights (The "Body")
    model = torchvision.models.resnet18(weights='DEFAULT')
    # Replace the "Head" to fit CIFAR-10 (10 classes instead of 1000)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(DEVICE)
    
    # 2. SETUP DATA
    # We use standard normalization for ResNet
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # download=False because you manually placed the file
    try:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    except RuntimeError:
        print("\n[Error] Dataset not found! Make sure 'cifar-10-python.tar.gz' is in the 'data' folder.")
        return

    # 3. SETUP OPTIMIZER
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    # 4. TRAINING LOOP
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        # Tqdm creates a nice progress bar
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()         # Clear previous gradients
            outputs = model(images)       # Forward pass
            loss = criterion(outputs, labels) # Calculate error
            loss.backward()               # Backward pass (gradients)
            optimizer.step()              # Update weights
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss/len(trainloader)})

    # 5. SAVE MODEL
    save_path = "resnet18_cifar10_trained.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n[Success] Trained model saved to: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    train_and_save()