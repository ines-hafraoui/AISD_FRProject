import torch
import os
from torchvision.models import resnet50

# 1. Create the folder
os.makedirs("weights", exist_ok=True)

# 2. Define the model (Matching buffalo_l r50)
model = resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 512)

# 3. Save it as a .pth file
torch.save(model.state_dict(), "weights/backbone.pth")
print("Success: Weights saved to weights/backbone.pth")