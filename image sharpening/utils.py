from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os

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