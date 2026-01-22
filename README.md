# UG-BAAR (Uncertainty-Gated Boundary Alignment Reprojection)

Official implementation of **UG-BAAR**, an uncertainty-aware boundary refinement framework for robust and calibrated brain tumor segmentation from MRI.

The proposed method integrates uncertainty-gated learning with boundary alignment and reprojection to improve segmentation accuracy, boundary delineation, and prediction reliability across heterogeneous MRI datasets.

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# Train the model
python train.py

# Evaluate a trained model
python evaluate.py --model checkpoints/best_model.pth
```

---

## 📁 Project Structure

```
UG-BAAR/
├── models/
│   ├── __init__.py
│   └── ug_baar.py
│
├── utils/
│   ├── __init__.py
│   ├── losses.py
│   ├── metrics.py
│   ├── baar_ops.py
│   └── seed.py
│
├── datasets/
│   ├── __init__.py
│   └── dataset.py
│
├── train.py
├── evaluate.py
├── config.yaml
├── requirements.txt
└── checkpoints/
```

---

## 🔧 Installation

1. Clone the repository:

```bash
git clone https://github.com/your-repo/UG-BAAR.git
cd UG-BAAR
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧠 Training

Train the UG-BAAR model using the default configuration:

```bash
python train.py
```

Training includes:

* Deterministic seed initialization
* Uncertainty-aware loss optimization
* Automatic checkpointing of the best-performing model

---

## 📊 Evaluation

Evaluate a trained model checkpoint:

```bash
python evaluate.py --model checkpoints/best_model.pth
```

The evaluation script reports Dice score and segmentation performance on the validation/test set.

---

## 🔁 Reproducibility

All results reported in the manuscript can be reproduced using the provided scripts.

To reproduce the experimental results:

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare the dataset according to your experimental protocol (e.g., BraTS or institutional MRI data).

3. Train the model:

```bash
python train.py
```

4. Evaluate the trained model:

```bash
python evaluate.py
```

Random seeds are fixed to ensure deterministic behavior across runs.

---

## 📄 Citation

If you use this code in your research, please cite the corresponding paper:

```
@article{UGBAAR2026,
  title={Uncertainty-Gated Boundary Alignment Reprojection for Robust Brain Tumor Segmentation},
  author={Indrakumar K, Ravikumar M},
  journal={Journal of Medical Systems (Springer)},
  year={2026}
}
```

---

## 📌 Notes

* This repository is intended for **research and academic use**.
* Dataset files are not included and must be obtained separately.
* The code is modular and can be extended to other medical image segmentation tasks.

