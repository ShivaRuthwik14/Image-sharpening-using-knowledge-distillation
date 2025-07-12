from models.teacher_model import TeacherSharpeningNet
from utils import SharpenDataset
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader

dataset = SharpenDataset("data/train")
loader = DataLoader(dataset, batch_size=16, shuffle=True)
model = TeacherSharpeningNet().cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    for blur, sharp in loader:
        blur, sharp = blur.cuda(), sharp.cuda()
        output = model(blur)
        loss = criterion(output, sharp)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: Loss={loss.item():.4f}")

torch.save(model.state_dict(), "teacher.pth")