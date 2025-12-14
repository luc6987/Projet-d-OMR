# GraphOMR: Optical Music Recognition Pipeline

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)

This project is an end-to-end Optical Music Recognition (OMR) pipeline that computationally reads handwritten music notation in documents, transcribing score images into ready-to-use MusicXML format.

The system leverages advanced deep learning object detection (YOLOv12) to extract basic musical symbols from score images, combines an MLP module to identify relationships between symbols, and implements a rule-based assembly module to resolve complex musical structure reconstruction issues including multi-staff grouping, polyphonic alignment, and tuplet detection.

The pipeline consists of four modules:
- **U-Net**: Staff line removal and image preprocessing (destaffing)
- **YOLO**: Musical symbol detection and classification using YOLOv12
- **MLP**: Link prediction between symbols to construct notation graphs
- **Assembler**: Rule-based semantic assembly and MusicXML generation

---

## 🚀 Quick Start

### 1. Environment Setup

Use `initialize.py` to automatically set up the environment and download raw data:

```bash
python initialize.py
```

This script will:
- Create a virtual environment in `.env/`
- Install all required dependencies from `requirements.txt`
- Download and prepare the MUSCIMA++ dataset

### 2. Activate Virtual Environment

After initialization, activate the virtual environment:

**On Linux/Mac:**
```bash
source .env/bin/activate
```

**On Windows:**
```bash
.env\Scripts\activate
```

### 3. Run the Pipeline

With the virtual environment activated, run the OMR pipeline:

```bash
# Run full pipeline (unet -> yolo -> mlp -> assembler)
python src/main.py pipeline

# Or run individual modules
python src/main.py train unet    # Train U-Net model
python src/main.py infer yolo    # Run YOLO inference
python src/main.py visualize mlp # Generate visualizations
python src/main.py stats assembler # Generate statistics
```

---

## ⚙️ Configuration

All pipeline parameters are configured in `setup.yml`. You can modify settings to customize:

- **Output paths**: Change `output_root`, `model_dir`, `vis_stat_dir`, etc.
- **Training parameters**: Batch size, learning rate, epochs, etc.
- **Module enable/disable**: Control which modules run in the pipeline
- **Inference settings**: Confidence thresholds, output formats, etc.

### Example: Modify Output Directory

```yaml
global:
  output_root: "my_output"  # Change output directory
  model_dir: "my_models"    # Change model directory
```

### Example: Adjust Training Parameters

```yaml
unet:
  train:
    batch_size: 8        # Increase batch size
    learning_rate: 0.001  # Adjust learning rate
    epochs: 200          # Train for more epochs

yolo:
  infer:
    confidence_threshold: 0.3  # Lower threshold for more detections
    output_dir: "${global.output_root}/yolo_results"
```

### Example: Enable/Disable Modules

```yaml
module_enable:
  unet: true
  yolo: true
  mlp: true
  assembler: false  # Disable assembler module
```

### Using Custom Config File

You can use a custom configuration file:

```bash
python src/main.py --config my_config.yml pipeline
```

---

## 📝 Usage Examples

```bash
# Train all models
python src/main.py train unet
python src/main.py train yolo
python src/main.py train mlp

# Run inference on a single module
python src/main.py infer unet
python src/main.py infer yolo

# Run complete pipeline (requires trained models)
python src/main.py pipeline

# Generate visualizations
python src/main.py visualize yolo
python src/main.py stats mlp
```

---

## 📁 Project Structure

```
Projet-d-OMR/
├── src/              # Source code
│   ├── main.py       # Main entry point
│   ├── unet/         # U-Net module
│   ├── yolo/         # YOLO module
│   ├── mlp/          # MLP module
│   ├── assembler/    # Assembler module
│   └── common/       # Common utilities
├── setup.yml         # Configuration file
├── initialize.py     # Environment setup script
├── requirements.txt  # Python dependencies
├── data/             # Dataset directory
├── model/            # Trained models
├── Output/           # Inference outputs
└── vis_stat/         # Visualization and statistics
```

---

## 📄 License

MIT License - see LICENSE file for details.
