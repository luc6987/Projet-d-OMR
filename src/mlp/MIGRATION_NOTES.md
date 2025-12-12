# MLP Linker Module Migration Notes

## Summary

All MLP linker code has been reorganized into `src/mlp/` with proper structure and conventions.

## Changes Made

### 1. Code Organization
- **Created**: `src/mlp/` directory with all MLP-related code
- **Moved**: MLP models from `model/model.py` to `src/mlp/model.py`
- **Created**: `src/mlp/train_core.py` - core training implementation
- **Created**: `src/mlp/train.py` - module entry point
- **Created**: `src/mlp/metrics.py` - validation and evaluation functions
- **Created**: `src/mlp/visualization.py` - training visualization
- **Created**: `src/mlp/analyze_dataset.py` - dataset statistics

### 2. Model Storage
- **Location**: `model/mlp/` - all MLP model weights stored here
- **Naming**: `{exp_name}_best.pth` (best model), `{exp_name}_ep{epoch}.pth` (checkpoints)
- **Format**: PyTorch checkpoints with model state_dict, optimizer, and config

### 3. Visualizations
- **Location**: `vis_stat/mlp/` - all training visualizations
- **Files**: `{exp_name}_training_metrics.png` - comprehensive training plots
- **Monitoring**: `src/viz_stat/mlp_linker_monitor.py` - analysis tool

### 4. Configuration Files
- **Location**: `src/mlp/assemblage/configs/assembler/` - all MLP configs
- **Created**: `MLP32_balanced.yaml` - balanced config
- **Created**: `MLP64_optimized.yaml` - larger model config
- **Existing**: `MLP32_optimized_f1.yaml` - optimized for F1 score

### 5. Documentation
- **Created**: `.cursorrules` - project conventions and guidelines
- **Created**: `src/mlp_linker/README.md` - module documentation

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

## Migration from Old Code

All MLP code has been consolidated into `src/mlp/`. Use:
- `src/main.py train mlp` for training (recommended)
- `src/mlp/train_core.py` for direct training script access
- Models are saved to `model/mlp/` 
- Visualizations go to `vis_stat/mlp/`

## Next Steps

1. Run dataset analysis to get statistics: `python src/mlp/analyze_dataset.py`
2. Choose appropriate config based on dataset size
3. Train with: `python src/main.py train mlp`
4. Monitor training with: `python src/viz_stat/mlp_linker_monitor.py`


