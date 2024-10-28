#!/usr/bin/env python
# -*- coding: utf-8 -*-

from modules.fix_random_seed import fix_random_seed
import albumentations as A
import numpy as np

# 変換の定義
transform32 = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0, rotate_limit=0, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.Cutout(num_holes=1, max_h_size=16, max_w_size=16, fill_value=0, p=0.2),
    A.RandomCrop(width=24, height=24, p=0.5)
])

resize32 = A.Compose([
    A.Resize(32, 32),
    A.ToFloat(max_value=255)
])

transform224 = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0, rotate_limit=0, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.Cutout(num_holes=1, max_h_size=16, max_w_size=16, fill_value=0, p=0.2),
    A.RandomCrop(width=24, height=24, p=0.5)
])


resize224 = A.Compose([
    A.Resize(224, 224),
    A.ToFloat(max_value=255),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 標準化
])


def apply_augmentation_with_original32(data, targets, num_times):
    fix_random_seed(0)
    augmented_data = []
    augmented_targets = []

    # 元のデータを追加
    for img in data:
        augmented_data.append(
            resize32(image=img)['image'])
    augmented_targets.extend(targets)

    for img, target in zip(data, targets):
        for _ in range(num_times):
            # 拡張を適用
            augmented_data.append(
                resize32(image=transform32(image=img)['image'])['image'])
            augmented_targets.append(target)

    return np.array(augmented_data), np.array(augmented_targets)


def apply_augmentation_with_original32_test(data):
    fix_random_seed(0)
    augmented_data = []

    # 元のデータを追加
    for img in data:
        augmented_data.append(
            resize32(image=img)['image'])

    return np.array(augmented_data)


def apply_augmentation_with_original224(data, targets, num_times):
    fix_random_seed(0)
    augmented_data = []
    augmented_targets = []

    # 元のデータを追加
    for img in data:
        augmented_data.append(
            resize224(image=np.repeat(img, 3, axis=-1))['image'])
    augmented_targets.extend(targets)

    for img, target in zip(data, targets):
        for _ in range(num_times):
            # 拡張を適用
            augmented_data.append(
                resize224(image=transform224(image=np.repeat(img, 3, axis=-1))['image'])['image'])
            augmented_targets.append(target)

    return np.array(augmented_data), np.array(augmented_targets)


def apply_augmentation_with_original224_test(data):
    fix_random_seed(0)
    augmented_data = []

    # 元のデータを追加
    for img in data:
        augmented_data.append(
            resize224(image=np.repeat(img, 3, axis=-1))['image'])

    return np.array(augmented_data)
