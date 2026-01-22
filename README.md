# Thanal

## Overview
Thanal is a project focused on virtual Near-Infrared (VNIR) estimation from RGB images of crop leaves. The primary aim is to predict diseases or stress in plants early, utilizing VNIR information without the need for actual NIR equipment.

## Current Status
We have developed a VNIR Estimation model based on U-Net with Attention mechanisms. This model has been trained on a dataset of RGB-NIR paired images specifically for capsicum.

## Demo
A Gradio interface is available for demonstrating the model's capabilities.

## Installation

## Setup

### 1. Create Virtual Environment
```bash
python -m venv thanal
```

### 2. Activate Environment

**Windows**

```bash
thanal\Scripts\activate
```

**macOS/Linux**

```bash
source thanal/bin/activate
```

### 3. Install Requirements

Create `requirements.txt`:

```txt
gradio
torch
torchvision
pillow
numpy
```

Install:

```bash
pip install -r requirements.txt
```

### 4. Verify (Optional)

```bash
python -c "import gradio, torch, torchvision, PIL, numpy"
```

## License
This project is licensed under the MIT License.

## Acknowledgments
Thanks to all contributors and datasets that made this project possible.