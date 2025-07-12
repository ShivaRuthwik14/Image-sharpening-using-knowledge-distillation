import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt

# Define Dataset Loader
class SharpenDataset(Dataset):
    def __init__(self, root_dir):
        self.blurred_paths = sorted(os.listdir(f"{root_dir}/blurred"))
        self.sharp_paths = sorted(os.listdir(f"{root_dir}/sharp"))
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.blurred_paths)

    def __getitem__(self, idx):
        blur_img = Image.open(f"{self.root_dir}/blurred/{self.blurred_paths[idx]}").convert('RGB')
        sharp_img = Image.open(f"{self.root_dir}/sharp/{self.sharp_paths[idx]}").convert('RGB')
        return self.transform(blur_img), self.transform(sharp_img)

# Define Models
class TeacherSharpeningNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1)
        )
    def forward(self, x):
        return self.net(x)

class StudentSharpeningNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1)
        )
    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    # Assuming the zip file is already extracted to image_sharpening_kd/data/train
    # If not, you'll need to manually extract it or add extraction code here

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = SharpenDataset("image_sharpening_kd/data/train")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Train the Teacher Model
    print("🧠 Training Teacher...")
    teacher = TeacherSharpeningNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(teacher.parameters(), lr=1e-3)

    for epoch in range(5):
        total_loss = 0
        for blur, sharp in loader:
            blur, sharp = blur.to(device), sharp.to(device)
            output = teacher(blur)
            loss = criterion(output, sharp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")

    torch.save(teacher.state_dict(), "teacher.pth")
    print("✅ Teacher model saved!")

    # Train the Student Model
    print("🧠 Training Student with KD...")
    student = StudentSharpeningNet().to(device)
    teacher.eval()

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    alpha, beta = 0.5, 0.5

    for epoch in range(5):
        total_loss = 0
        for blur, sharp in loader:
            blur, sharp = blur.to(device), sharp.to(device)
            with torch.no_grad():
                teacher_out = teacher(blur)
            student_out = student(blur)

            loss_gt = criterion(student_out, sharp)
            loss_kd = criterion(student_out, teacher_out)
            loss = alpha * loss_gt + beta * loss_kd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: KD Loss = {total_loss/len(loader):.4f}")

    torch.save(student.state_dict(), "student.pth")
    print("✅ Student model saved!")

    # Visualize results
    print("📊 Visualizing results...")
    teacher = TeacherSharpeningNet().to(device)
    teacher.load_state_dict(torch.load("teacher.pth"))
    teacher.eval()

    student = StudentSharpeningNet().to(device)
    student.load_state_dict(torch.load("student.pth"))
    student.eval()

    for i in range(len(dataset)):
        blurred_img, sharp_img = dataset[i]
        input_tensor = blurred_img.unsqueeze(0).to(device)

        with torch.no_grad():
            teacher_out = teacher(input_tensor).squeeze(0).cpu()
            student_out = student(input_tensor).squeeze(0).cpu()

        # Convert to PIL
        blur_pil = TF.to_pil_image(blurred_img)
        teacher_pil = TF.to_pil_image(teacher_out.clamp(0, 1))
        student_pil = TF.to_pil_image(student_out.clamp(0, 1))
        sharp_pil = TF.to_pil_image(sharp_img)

        # Plot
        titles = ["Blurred Input", "Teacher Output", "Student Output", "Ground Truth"]
        images = [blur_pil, teacher_pil, student_pil, sharp_pil]

        plt.figure(figsize=(16, 4))
        for j in range(4):
            plt.subplot(1, 4, j + 1)
            plt.imshow(images[j])
            plt.title(titles[j])
            plt.axis("off")
        plt.suptitle(f"Sample {i}", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"sample_{i}_sharpening_results.png") # Save figures
        plt.close() # Close figures to prevent display issues in some environments
    print("✅ Results visualizations saved as PNG files.")
