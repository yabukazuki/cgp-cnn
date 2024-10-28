#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import random


def fix_random_seed(seed: int):
    """再現性の確保

    乱数シードを固定する

    Args:
        seed (int): シード値
    """

    random.seed(seed)
    np.random.seed(seed)

    # 初期重み
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

    g = torch.Generator()
    g.manual_seed(seed)
