# MLP Linker Module

This module contains all code for training and using MLP models to link musical notation symbols.

## Structure

- `model.py`: MLP model definitions
- `train.py`: Main training script
- `metrics.py`: Evaluation and validation functions
- `visualization.py`: Training visualization functions
- `analyze_dataset.py`: Dataset statistics analysis

## Usage

### Training

```bash
python src/main.py train mlp
```

### Testing

```bash
python src/mlp/train_core.py --test_only --exp_name my_experiment --classes essential
```

### Dataset Analysis

```bash
python src/mlp/analyze_dataset.py --class_schema essential
```

## Output Locations

- **Model weights**: `model/mlp/{exp_name}/model_best.pth`
- **Checkpoints**: `model/mlp/{exp_name}/model_ep{epoch}.pth`
- **Visualizations**: `vis_stat/mlp/{exp_name}_training_metrics.png`

## Configuration

See `src/mlp/assemblage/configs/assembler/` for configuration files:
- `default.py`: Base configuration
- `MLP32_balanced.yaml`: Balanced config for medium datasets
- `MLP64_optimized.yaml`: Larger model for big datasets


