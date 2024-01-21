# Author: ray
# Date: 1/22/24
# Description:

import numpy as np
import torch
import random


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Check if MPS is available for Apple's M1 chips
    if torch.backends.mps.is_available():
        # MPS-specific seed settings can be added here if available
        pass

    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


set_seed(87)
