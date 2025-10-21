from yacs.config import CfgNode as CN


_C = CN()

# System setup
_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 0
_C.SYSTEM.NUM_WORKERS = 0
_C.SYSTEM.SEED = 95

# Data setup
_C.DATA = CN()
_C.DATA.DATA_CONFIG = "configs/muscima_nogrammar.yaml"

# Model setup
_C.MODEL = CN()
_C.MODEL.MODE = "MLP"
_C.MODEL.MLP_CONFIG = [32, 32]
_C.MODEL.EMBEDDING_DIM = 32
_C.MODEL.VOCAB_DIM = 177
_C.MODEL.MLP_PARA = "configs/assembler/MLP32_balanced.yaml"


# DATA PATH
_C.PATH = CN()
_C.GT_ROOT = 'data/MUSCIMA++/v2.0/data/annotations'
_C.IM_ROOT = 'data/MUSCIMA++/datasets_r_staff/images'
_C.SPFILE = 'splits/mob_split.yaml'

# Training parameter
_C.TRAIN = CN()
_C.TRAIN.NUM_EPOCHS = 10
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.EVAL_FREQUENCY = 5
_C.TRAIN.SAVE_FREQUENCY = 5
# Loss and Optimizer
_C.TRAIN.POS_WEIGHT = 1
_C.TRAIN.OPTIMIZER = "AdamW"
_C.TRAIN.LEARNING_RATE = 1e-3



# Inference/Evaluation setup
_C.EVAL = CN()
_C.EVAL.BATCH_SIZE = 32

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`
