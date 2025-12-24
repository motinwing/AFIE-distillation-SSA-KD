# SSA-KD: Self-Structure-Aware Knowledge Distillation

This repository contains the official implementation of the paper **"SSA-KD: Self-Structure-Aware Knowledge Distillation for Convolutional Neural Networks"**.

The project introduces a novel knowledge distillation framework that integrates a structure-aware pruning method to automatically derive a student model from the teacher model, ensuring structural compatibility and higher efficiency.

---

## 📁 Repository Structure

- **`pruneutil.py`** – Contains core pruning utilities, including:
  - Pruning functions for convolutional, normalization, and fully connected layers.
  - Criteria functions for filter importance evaluation (Taylor and AFIE-based).
  - Index selection and layer-wise pruning logic.

- **`resnet.py`** – Defines ResNet models (ResNet-18, 34, 50, 101) with built-in pruning support.
  - Implements `BasicBlock` and `Bottleneck` modules.
  - Includes `resnet_prune()` method for structured pruning.

- **`main.py`** – Main script for training, pruning, and distillation experiments.
  - Supports datasets: CIFAR-100, ImageNet-100, ImageNet-1K, Flowers.
  - Implements teacher training, pruning, and knowledge distillation pipelines.
  - (Note: Federated learning code is present but unused in this project.)

- **`client.py`** – Defines the `PreparedModel` class for model training, distillation, and evaluation.
  - Handles data loading, optimization, learning rate scheduling, and pruning.
  - Supports both standard training and distillation with temperature scaling.

- **`mongo.py`** & **`draw_final.py`** – Auxiliary files for visualization and multi-threading (used for plotting accuracy curves).

---

## 🛠️ Requirements

- Python 3.8+
- PyTorch 1.9+
- torchvision
- matplotlib
- thop (for FLOPs calculation)
- Optional: CUDA-capable GPU for acceleration

Install dependencies via:

```bash
pip install torch torchvision matplotlib thop
```

---

## 🚀 Usage

### 1. Train a Teacher Model

To train a ResNet-50 teacher on ImageNet-100:

```bash
python main.py
```

Set `trained = False` in `main.py` for training from scratch, or `True` to load a pre-trained model.

### 2. Prune the Teacher Model

In `main.py`, call `prune_test(server)` to apply structured pruning with a specified pruning rate. The pruning method combines Taylor pruning and weight pruning approaches.

### 3. Perform Knowledge Distillation

Call `distillation_test(server)` to distill knowledge from the teacher to the pruned student model. Adjust temperature `T` and pruning rate `PR` as needed.

---

## ⚙️ Key Parameters

- `T` – Temperature for distillation (default: 3)
- `PR` – Pruning rate (default: 0.5)
- `theta` – Weighting factor between Taylor pruning and weight pruning criteria (range: 0-1)
- `total_epochs` – Training epochs (default: 150)

---

## 📈 Results

As reported in the paper (available upon request or through academic publication), SSA-KD achieves:

- Higher compression rates with minimal accuracy drop.
- Efficient student model customization via structured pruning.

---

## 📄 Citation

If you use this code or the SSA-KD method in your research, please cite:

```bibtex
@article{lu2024ssakd,
  title={SSA-KD: Self-Structure-Aware Knowledge Distillation for Convolutional Neural Networks},
  author={Lu, Yiheng and Zhang, Zhihui and Guan, Ziyu and Zhao, Wei and Yang, Yaming and Xu, Cai},
  journal={arXiv preprint},
  year={2024}
}
```

---

## 📧 Contact

For questions or issues, please open an issue in this repository.

---

**License:** This project is for academic use only. Please refer to the paper for terms and conditions.
