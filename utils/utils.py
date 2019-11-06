import os
import sys

import torch
import importlib
import shutil
import logging
import pickle
import random
import numpy as np
import json


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


level5data = None

