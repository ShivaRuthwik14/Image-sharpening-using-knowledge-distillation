from models.student_model import StudentSharpeningNet
from models.teacher_model import TeacherSharpeningNet
from utils import SharpenDataset
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader

dataset = SharpenDataset("data/train")
loader = DataLoader(dataset, batch_size=16, shuffle=True)

teacher = TeacherSharpeningNet().cuda()
teacher.load_state_dict(torch.load("teacher.pth"))
teacher.eval()

student = StudentSharpeningNet().cuda()
criterion_mse = nn.MSELoss()

optimizer = optim.Adam(student.parameters(), lr=1e-3)

α, β = 0.5, 0.5

for epoch in range(10):
    for blur, sharp in loader:
        blur, sharp = blur.cuda(), sharp.cuda()
        with torch.no_grad():
            teacher_out = teacher(blur)
        student_out = student(blur)

        loss_gt = criterion_mse(student_out, sharp)
        loss_distill = criterion_mse(student_out, teacher_out)
        loss = α * loss_gt + β * loss_distill

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: Loss={loss.item():.4f}")

torch.save(student.state_dict(), "student.pth")