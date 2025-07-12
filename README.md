# 🔍 Image Sharpening using Knowledge Distillation

This repository demonstrates an approach to **image sharpening** using **Knowledge Distillation (KD)**. A deep neural network is trained to convert blurred images into sharpened ones using a pre-trained teacher network to guide a smaller, efficient student model.

---

## 📁 Project Structure

```
├── dataset/
│   ├── blurred/
│   └── sharp/
├── models/
│   ├── teacher_model.h5
│   └── student_model.h5
├── student_training.py
├── evaluate_models.py
├── utils.py
└── sharpen_kd_colab.ipynb
```

---

## 🚀 How to Run (Google Colab)

### 🔗 Open in Colab:
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShivaRuthwik14/Image-sharpening-using-knowledge-distillation/blob/main/sharpen_kd_colab.ipynb)

### 📦 Dependencies

All dependencies are included in the notebook, but here’s a quick list:
```bash
pip install tensorflow numpy matplotlib opencv-python
```

---

## 🧠 Approach

1. **Teacher Network**: A deep CNN trained on paired blurred and sharp images.
2. **Student Network**: A lightweight CNN trained to mimic the teacher’s output using knowledge distillation loss (combination of L2 loss and teacher-student feature loss).
3. **Evaluation**: PSNR and SSIM scores are used to compare:
   - Input blurred image
   - Output of student model
   - Ground truth sharp image

---

## 🗃️ Dataset

- Place your dataset under the `dataset/` folder:
  - `dataset/blurred/` – Input blurred images
  - `dataset/sharp/` – Ground truth sharpened images

Modify the paths in `student_training.py` or the notebook if necessary.

---

## 📊 Output

The notebook will:
- Train the student network using KD
- Visualize before vs after sharpening
- Plot PSNR and SSIM scores

---

## ✅ Results Example

| Input (Blurred) | Student Output (Sharp) | Ground Truth |
|------------------|------------------------|---------------|
| ![Blur](samples/blur1.png) | ![Sharp](samples/student1.png) | ![GT](samples/sharp1.png) |

---

## 📬 Contact

Created by [Shiva Ruthwik](https://github.com/ShivaRuthwik14)  
Feel free to open issues or pull requests if you'd like to contribute!

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
