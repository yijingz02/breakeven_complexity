from nsm.typing import *
from configs.flow import default

def get_config():
  cfg = default()
  cfg.file = __file__

  cfg.path = "newflow.NpyFlowVorticity"

  cfg.config = ConfigDict({
    "base_dir": placeholder(str),
    "train_glob": "train/data_*.npy",
    "val_glob":   "val/data_*.npy",
    "test_glob":  "val/data_*.npy",
  })

  return cfg
