# Link Prediction Model Recommendations

## Problem Analysis

You have **10+ annotated XML files** with Outlinks relationships between musical symbols. This is a **binary link prediction problem** where you need to predict whether two objects should be connected.

**Input Features:**
- Source object: class name, bounding box (top, left, width, height)
- Target object: class name, bounding box (top, left, width, height)
- Spatial relationships: distance, relative position, overlap, etc.

**Output:**
- Binary classification: link exists (1) or not (0)

## Recommended Models

### 1. **MLP (Multi-Layer Perceptron)** ⭐ **Current Project Model**

**Pros:**
- ✅ Already implemented in your project (`src/mlp/model.py`)
- ✅ Simple and fast training/inference
- ✅ Works well with structured features (bbox + class embeddings)
- ✅ Good for small to medium datasets (10-100 images)
- ✅ Easy to interpret and debug

**Cons:**
- ❌ Doesn't capture global graph structure
- ❌ Limited to pairwise relationships
- ❌ May miss complex multi-hop dependencies

**Best for:** Your current use case (10+ images, pairwise link prediction)

**Architecture:**
```
Input: [source_bbox(4) + source_class_embed(d) + target_bbox(4) + target_class_embed(d)]
  ↓
MLP Layers (e.g., [128, 64, 32])
  ↓
Output: Binary classification (link probability)
```

### 2. **Graph Neural Network (GNN)** ⭐ **For Complex Relationships**

**Pros:**
- ✅ Captures global graph structure
- ✅ Can model multi-hop relationships
- ✅ Better for complex musical notation patterns
- ✅ Project already has `torch-geometric` installed

**Cons:**
- ❌ More complex to implement
- ❌ Requires more data (50+ images recommended)
- ❌ Slower training/inference
- ❌ Harder to debug

**Best for:** Large datasets (50+ images) with complex relationships

**Architecture Options:**
- **GCN (Graph Convolutional Network)**: Basic graph convolution
- **GAT (Graph Attention Network)**: Attention-based, better for variable neighborhoods
- **GraphSAGE**: Inductive learning, good for large graphs
- **Edge Prediction GNN**: Specifically designed for link prediction

**Example Architecture:**
```python
# Node features: [class_embed, bbox_normalized]
# Edge features: [spatial_features, class_pair_features]
GNN Layers → Edge Classifier → Link Probability
```

### 3. **Transformer-based Link Predictor**

**Pros:**
- ✅ State-of-the-art performance
- ✅ Can model long-range dependencies
- ✅ Attention mechanism captures important relationships

**Cons:**
- ❌ Very complex
- ❌ Requires large datasets (100+ images)
- ❌ Slow training
- ❌ Overkill for your current dataset size

**Best for:** Very large datasets with complex patterns

## Recommendation for Your Case (10+ Images)

### **Primary Recommendation: Enhanced MLP** ⭐

Given your dataset size (10+ images), I recommend **sticking with MLP but enhancing it**:

1. **Keep current MLP architecture** (already works)
2. **Add spatial features**:
   - Distance between centers
   - Relative position (above/below, left/right)
   - Overlap ratio
   - Angle/direction
   - Size ratio

3. **Improve data augmentation**:
   - Random crop/zoom
   - Horizontal flips (for music sheets)
   - Noise injection

4. **Better negative sampling**:
   - Hard negative mining (focus on difficult cases)
   - Balanced positive/negative ratio

### **Secondary Option: Lightweight GNN**

If you want to experiment with GNNs:
- Use **GAT (Graph Attention Network)** with 2-3 layers
- Keep it simple: node features = [class_embed, normalized_bbox]
- Edge features = [distance, relative_position]
- Train on all 10+ images together

## Implementation Plan

### Phase 1: Data Preparation (Current Priority)

1. **Collect all XML files**:
   ```bash
   # Find all XML annotation files
   find data/v1.0/data/cropobjects_withstaff -name "*.xml" > xml_list.txt
   ```

2. **Extract link pairs**:
   - For each XML file, extract all (source, target) pairs from Outlinks
   - Create positive samples: all existing links
   - Create negative samples: random pairs without links (balanced)

3. **Feature extraction**:
   - Normalize bounding boxes (relative to image size)
   - Compute spatial features (distance, angle, overlap)
   - Encode class names as embeddings

4. **Split dataset**:
   - Train: 70% of images
   - Validation: 15% of images
   - Test: 15% of images

### Phase 2: Model Training

**Option A: Use Existing MLP (Recommended)**
```bash
# Train with your 10+ XML files
python src/mlp/train.py
```

**Option B: Enhanced MLP**
- Modify `src/mlp/model.py` to add spatial features
- Update data loading to include computed features

**Option C: GNN Implementation**
- Create new module `src/gnn/`
- Implement GAT-based link predictor
- Train on graph-structured data

### Phase 3: Evaluation

- **Metrics**: Precision, Recall, F1-score
- **Visualization**: Show predicted links vs ground truth
- **Error analysis**: Which link types are hardest to predict?

## Data Requirements

### Minimum for MLP:
- ✅ **10+ images** (you have this)
- ✅ **~1000+ link pairs** (you likely have this)
- ✅ **Balanced positive/negative samples**

### Minimum for GNN:
- ⚠️ **50+ images** (you need more)
- ⚠️ **~5000+ link pairs**
- ⚠️ **More complex relationships**

## Quick Start: Train with Your Data

### Step 1: Prepare Data Split
```python
# Create train/val/test split file
# File: data/v1.0/splits/custom_split.txt
# Format: one image name per line
p001
p002
...
```

### Step 2: Train MLP
```bash
python src/mlp/train.py \
    --gt_annotations_root data/v1.0/data/cropobjects_withstaff \
    --images_root data/v1.0/data/images \
    --split_file data/v1.0/splits/custom_split.txt
```

### Step 3: Evaluate
```bash
python src/mlp/infer.py \
    --model_path model/mlp/your_experiment/model_best.pth \
    --test_data ...
```

## Model Comparison Table

| Model | Dataset Size | Training Time | Accuracy | Complexity |
|-------|-------------|---------------|----------|------------|
| **MLP** | 10+ images | Fast (min) | Good | Low |
| **Enhanced MLP** | 10+ images | Fast (min) | Better | Medium |
| **GNN (GAT)** | 50+ images | Medium (hours) | Best | High |
| **Transformer** | 100+ images | Slow (days) | Excellent | Very High |

## Next Steps

1. **Immediate**: Use existing MLP with your 10+ XML files
2. **Short-term**: Enhance MLP with spatial features
3. **Long-term**: If you get 50+ images, consider GNN

## Code Structure Suggestion

```
src/
├── mlp/              # Current MLP (keep and enhance)
│   ├── model.py      # Add spatial features
│   └── train.py      # Already works
│
└── gnn/              # Future GNN implementation (optional)
    ├── model.py      # GAT-based link predictor
    ├── train.py
    └── data.py       # Graph data preparation
```

## References

- **MLP for Link Prediction**: Your current implementation
- **GNN for Link Prediction**: 
  - Graph Attention Networks (GAT)
  - GraphSAGE
  - PyTorch Geometric documentation
- **Music Notation Link Prediction**:
  - Research papers on OMR graph models
  - MUSCIMA++ dataset papers


