# GraphOMR: A YOLOv8 and GNN Pipeline for Handwritten Music Recognition

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![Status](https://img.shields.io/badge/status-in--development-orange)

> An end-to-end Optical Music Recognition (OMR) pipeline that leverages state-of-the-art object detection with YOLOv8 and the relational power of Graph Neural Networks (GNNs) to transcribe handwritten music sheets into a digital format.

This project tackles the complex task of reading handwritten music, which is challenging due to the variability in handwriting and the complex, two-dimensional relationships between musical symbols. Our approach first identifies musical primitives and then uses a graph-based model to understand their musical context and reconstruct the score.

---
## 📜 Table of Contents
* [Introduction](#-introduction)
* [How It Works: The Pipeline](#-how-it-works-the-pipeline)
* [Features](#-features)
* [Getting Started](#-getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#-usage)
* [Training Your Own Models](#-training-your-own-models)
* [Roadmap](#-roadmap)
* [Contributing](#-contributing)
* [License](#-license)
* [Acknowledgments](#-acknowledgments)

---
## 🎵 Introduction

[cite_start]Optical Music Recognition (OMR) aims to teach computers how to read music notation[cite: 6]. While commercial OMR tools exist for printed scores, handwritten music remains a largely unsolved problem. [cite_start]The core challenges lie in the **contextual nature of music notation**—where symbols like clefs and key signatures affect distant notes—and the **ubiquitous two-dimensional spatial relationships** that define chords, melodies, and voices[cite: 268, 290].

**GraphOMR** addresses this by treating a music sheet not as a sequence of symbols, but as a graph. This allows us to explicitly model the vertical (harmony), horizontal (melody), and logical (contextual) relationships that a human musician intuitively understands.

---
## 🛠️ How It Works: The Pipeline

Our pipeline is a multi-stage process that moves from raw pixels to a structured, symbolic representation of the music.

![OMR pipeline flowchart](https://i.imgur.com/example-pipeline-image.png) #### ### 1. Music Object Detection (YOLOv8)
The first step is to locate and classify all the primitive musical symbols on the scanned sheet music. [cite_start]This stage aligns with the "Music Object Detection" phase of a traditional OMR pipeline[cite: 620]. We use a **YOLOv8** model trained on a custom dataset of handwritten musical symbols.

* **Input**: A scanned image of a handwritten music sheet.
* **Process**: The YOLOv8 model scans the image and outputs bounding boxes for each detected symbol primitive (e.g., `notehead-black`, `stem`, `quarter-rest`, `g-clef`, `accidental-sharp`).
* **Output**: A list of detected symbol classes and their coordinates.

#### ### 2. Graph Construction
This is the crucial bridge between the vision and the reasoning components. The detected symbols are converted into a graph structure $G = (V, E)$, where symbols are nodes and their relationships are edges.

* **Nodes (V)**: Each symbol detected by YOLOv8 becomes a node in the graph. Node features include the symbol's class, bounding box, and optionally, a visual feature vector.
* **Edges (E)**: Edges are created based on spatial heuristics to represent musical relationships. Each edge has features, such as the **relative displacement vector $(\Delta x, \Delta y)$** between symbols. Common edge types include:
    * `follows`: Connects sequential symbols.
    * `is_simultaneous_with`: Connects vertically aligned noteheads in a chord.
    * `is_part_of`: Connects primitives of a single note (e.g., `notehead` to `stem`).
    * `modifies`: Connects contextual symbols (like a clef) to the notes they affect.

#### ### 3. Semantic Reconstruction (GNN)
Once the initial graph is built, a **Graph Neural Network (GNN)** is used to perform **semantic reconstruction**. [cite_start]This corresponds to the "Notation Assembly" stage where relationships between symbols are recovered to understand the musical notation[cite: 621]. The GNN propagates information across the graph, allowing it to learn the musical context and refine the initial, fragmented data.

* **Process**: The GNN takes the graph as input and performs tasks like:
    * **Node Label Refinement**: Correcting misclassifications from YOLOv8 by using the context of neighboring nodes. For example, it can learn that a `notehead-black` connected to a `stem` and a `beam` is an `eighth-note`.
    * **Relationship Prediction**: Classifying edges to identify musical structures like slurs, ties, or tuplets.
* **Output**: A semantically enriched graph where nodes represent complete musical objects (e.g., a C4 quarter note) and edges represent confirmed musical relationships.

#### ### 4. Digital Score Generation
[cite_start]The final step, called "Encoding" in the traditional pipeline, is to traverse the enriched graph and convert it into a standard digital format[cite: 623].

* **Process**: A deterministic parser reads the graph and assembles the musical information. It follows melodic lines, builds chords, and applies the rules from clefs, keys, and accidentals.
* [cite_start]**Output**: A **MusicXML** file for editing in notation software or a **MIDI** file for playback[cite: 623, 624].

---
## ✨ Features
* **Handwritten Support**: Specifically designed for the challenges of handwritten scores.
* **End-to-End Pipeline**: From image to digital score with a single command.
* **State-of-the-Art Models**: Utilizes the speed and accuracy of YOLOv8 for detection and the contextual power of GNNs for musical understanding.
* **Extensible**: Both the detector and the GNN can be retrained on custom datasets for different musical styles or notations.
* **Standard Output**: Generates MusicXML, a widely supported format for digital sheet music.

---
## 🚀 Getting Started

### Prerequisites
* Python 3.11+
* PyTorch
* `ultralytics` (for YOLOv8)
* `torch-geometric` (for GNNs)
* OpenCV

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/luc6987/Projet-d-OMR.git](https://github.com/luc6987/Projet-d-OMR.git)
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the pre-trained models:**
    ```bash
    # (Add command to download models from a release or cloud storage)
    ./download_models.sh
    ```

---
## 📁 Data Structure

The project uses the MUSCIMA++ dataset for training and evaluation. The `data/` directory structure is organized as follows:

```
data/
└── MUSCIMA++/
    ├── v2.0/
    │   ├── data/
    │   │   └── annotations/          # 140 XML annotation files
    │   │       └── CVC-MUSCIMA_W-*_N-*_D-ideal.xml
    │   └── specifications/
    │       ├── cvc-muscima-image-list.txt
    │       ├── testset-dependent.txt
    │       ├── testset-independent.txt
    │       └── mff-muscima-mlclasses-annot.xml
    ├── datasets_r_staff_essn_crop/   # Processed dataset (staff removed, cropped)
    │   ├── train/
    │   │   ├── images/               # Training images
    │   │   └── labels/               # YOLO format labels
    │   ├── valid/
    │   │   ├── images/               # Validation images
    │   │   └── labels/               # YOLO format labels
    │   └── test/
    │       ├── images/               # Test images
    │       └── labels/               # YOLO format labels
    └── datasets_r_staff/
        └── images/                   # 140 PNG images (staff removed)
            └── CVC-MUSCIMA_W-*_N-*_D-ideal.png
```

**Dataset Statistics:**
- Total files: ~3,487 files
- Total directories: 18 directories
- Training/Validation/Test splits are pre-processed and ready for YOLO training

**Data Sources:**
- **MUSCIMA++ v2.0**: Original annotations and specifications
- **Preprocessed datasets**: Staff-removed versions with YOLO-compatible labels for object detection

---
## 💻 Usage

To transcribe a handwritten music sheet, run the `recognize.py` script:

```bash
python recognize.py --input /path/to/your/sheet.png --output /path/to/output/score.xml
