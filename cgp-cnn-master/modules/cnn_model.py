#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from typing import Tuple, List


# 畳み込み層 (初期宣言で入力チャネルは定義しない)
class MyConv(nn.Module):
    """畳み込み層

    Attributes:
        out_channels (int): 出力チャネル数
        kernel_size (int): カーネルサイズ
        padding (int): パディングサイズ

        conv (nn.Conv2d): 畳み込み層
        set_weight (bool): True なら重みを設定済み

        device (str): cuda or cpu
    """

    def __init__(self, out_channels: int, kernel_size: int, padding: int, device: str):
        """コンストラクタ

        Args:
            out_channels (int): 出力チャネル数
            kernel_size (int): カーネルサイズ
            padding (int): パディングサイズ
            device (str): cuda or cpu.
        """
        super(MyConv, self).__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv = None
        self.set_weight = False

        self.device = device

    def forward(self, x):
        if self.set_weight:
            return self.conv(x)
        else:
            in_channels = x.shape[1]  # 入力チャネル数の取得

            self.conv = nn.Conv2d(
                in_channels=in_channels, out_channels=self.out_channels,
                kernel_size=self.kernel_size, stride=1, padding=self.padding)

            init.kaiming_normal_(
                self.conv.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(self.conv.bias, 0.0)

            # self.conv.cuda() がないと nn.module の継承ができず, 訓練がされない
            self.conv.to(self.device)
            self.set_weight = True

            return self.conv(x)


class ConvBlock(nn.Module):

    """畳み込みブロック

    CONV -> Batch -> DropOut (有無) -> ReLU

    Attributes:
        padding (int): パディングサイズ
        out_channels (int): 出力チャネル数

        conv1 (MyConv): 畳み込み層
        bn1 (nn.BatchNorm2d): バッチノーマライゼーション
        dropout1 (nn.Dropout2d): ドロップアウト

    """

    def __init__(self, kernel_size: int, out_channels: int, device: str, rate_dropout: float):
        """コンストラクタ

        Args:
            kernel_size (int): バッチサイズ
            out_channels (int): 出力チャネル数
            device (str): cuda or cpu
            rate_dropout (float): ドロップアウト率
        """
        super(ConvBlock, self).__init__()
        self.padding = kernel_size // 2
        self.out_channels = out_channels

        self.conv1 = MyConv(out_channels=out_channels, kernel_size=kernel_size,
                            padding=self.padding, device=device)

        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        nn.init.constant_(self.bn1.weight, 1.0)
        nn.init.constant_(self.bn1.bias, 0.0)

        self.dropout1 = nn.Dropout2d(p=rate_dropout)

    def forward(self, x):
        """_summary_

        # CONV -> Batch -> DropOut -> ReLU

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        num_params = 0

        for name, f in self.named_children():
            if 'conv1' in name:
                x = f(x)
                # パラメータ数 = 出力チャネル数 * 入力チャネル数 * カーネル高さ * カーネル幅 + 出力チャネル数 (bias)
                num_params += (f.conv.weight.shape[0] * f.conv.weight.shape[1] *
                               f.conv.weight.shape[2] * f.conv.weight.shape[3] + f.conv.weight.shape[0])
            elif 'bn1' in name:
                x = f(x)
                num_params += f.weight.shape[0] + f.bias.shape[0]
        return (F.relu(self.dropout1(x)), num_params)


class ResBlock(nn.Module):
    """残差ブロック

    ReLU((CONV -> Batch -> DropOut (有無) -> ReLU -> CONV -> Batch) + (x))

    Attributes:

    """

    def __init__(self, kernel_size: int, out_channels: int,
                 rate_dropout: float, device: str):
        """コンストラクタ

        Args:
            kernel_size (int): カーネルサイズ
            out_channels (int): 出力チャネル数
            rate_dropout (float): ドロップアウト率
            device (str): cuda or cpu.

        """
        super(ResBlock, self).__init__()
        self.padding = kernel_size // 2
        self.out_channels = out_channels

        self.device = device

        # CONV -> Batch -> ReLU
        self.conv1 = MyConv(out_channels=out_channels, kernel_size=kernel_size,
                            padding=self.padding, device=self.device)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.dropout1 = nn.Dropout2d(p=rate_dropout)
        self._act1 = nn.ReLU()

        self.conv2 = MyConv(out_channels=out_channels, kernel_size=kernel_size,
                            padding=self.padding, device=self.device)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        nn.init.constant_(self.bn1.weight, 1.0), nn.init.constant_(
            self.bn1.bias, 0.0)
        nn.init.constant_(self.bn2.weight, 1.0), nn.init.constant_(
            self.bn2.bias, 0.0)

    def forward(self, x, h):
        num_params = 0
        for name, f in self.named_children():
            if 'conv' in name:
                x = f(x)
                num_params += (f.conv.weight.shape[0] * f.conv.weight.shape[1] *
                               f.conv.weight.shape[2] * f.conv.weight.shape[3] + f.conv.weight.shape[0])
            elif 'bn' in name:
                x = f(x)
                num_params += f.weight.shape[0] + f.bias.shape[0]
            elif 'act' in name:
                x = f(x)
            elif 'dropout' in name:
                x = f(x)

        in_data = [x, h]
        in_data = adjust_image_channel_size(
            in_data=in_data, device=self.device)

        return (F.relu(in_data[0]+in_data[1]), num_params)


class MyCat(nn.Module):
    """画像をチャネル方向に結合

    チャネル方向に結合, チャネルが増える

    """

    def __init__(self, dim: int = 1):
        """コンストラクタ

        Args:
            dim (int, optional): 軸. Defaults to 1.
        """
        super(MyCat, self).__init__()
        self.dim = dim

    def forward(self, xs: Tuple):

        return torch.cat(xs, dim=self.dim)


def adjust_image_map_size(in_data):
    """画像サイズの調整

    画像サイズの調整, 最大値プーリングによって小さい方に合わせる

    Args:
        in_data (_type_): _description_

    Returns:
        _type_: _description_
    """

    small_in_id, large_in_id = (
        0, 1) if in_data[0].shape[2] < in_data[1].shape[2] else (1, 0)
    pool_num = torch.floor(
        torch.log2(torch.tensor(in_data[large_in_id].shape[2] / in_data[small_in_id].shape[2])))
    for _ in torch.arange(pool_num):
        in_data[large_in_id] = F.max_pool2d(
            input=in_data[large_in_id], kernel_size=2, stride=2, padding=0, ceil_mode=False)

    return in_data


def adjust_image_channel_size(in_data, device):
    """チャネルサイズの調整

    チャネルサイズの調整, ゼロパディングによって大きい方に合わせる

    Args:
        in_data (_type_): _description_

    Returns:
        _type_: _description_
    """

    small_ch_id, large_ch_id = (
        0, 1) if in_data[0].shape[1] < in_data[1].shape[1] else (1, 0)
    nums_pad = int(in_data[large_ch_id].shape[1] -
                   in_data[small_ch_id].shape[1])
    in_data[small_ch_id] = torch.cat(
        (in_data[small_ch_id], torch.full((in_data[small_ch_id].shape[0], nums_pad, in_data[small_ch_id].shape[2], in_data[small_ch_id].shape[3]), fill_value=0, device=device)), dim=1)
    return in_data


class MyFullyConnectedBlock(nn.Module):
    """線形層

    線形層（全結合層を想定）

    Attributes:

        out_features (int): 出力数 (クラス数)
        device (str): cuda or cpu

        in_features (Union[int, None]): 入力数. Defaults to None
        linear (Union[nn.Linear, None]): 線形層. Defaults to None
        set_weight (bool): True なら線形層に重みを設定済み

    """

    def __init__(self, out_features: int, device: str):
        """コンストラクタ

        Args:
            out_features (int): 出力数
            device (str): cuda or cpu
        """
        super(MyFullyConnectedBlock, self).__init__()
        self.out_features = out_features
        self.device = device

        self.in_features = None
        self.linear = None
        self.set_weight = False

    def forward(self, x: list):
        # 平滑化
        x = torch.cat(
            list(map(lambda y: torch.flatten(F.avg_pool2d(y, kernel_size=y.size()[3]), start_dim=1), x)), dim=1)

        if self.set_weight:
            return self.linear(x)
        else:
            self.in_features = x.shape[1]
            self.linear = nn.Linear(self.in_features, self.out_features)
            init.kaiming_normal_(self.linear.weight,
                                  mode='fan_in', nonlinearity='relu')
            init.zeros_(self.linear.bias)
            self.linear.to(self.device)
            self.set_weight = True
        return self.linear(x)


# Construct a CNN model using CGP (list)
class CGP2CNN(nn.Module):
    def __init__(self, cgp, n_class, device, rate_dropout, search_space_obj):
        super(CGP2CNN, self).__init__()
        self.cgp = cgp
        self.search_space_obj = search_space_obj
        self.device = device

        i = 1
        for name, in1, in2 in self.cgp:
            if name == 'pool_max':
                setattr(self, '_'+name+'_'+str(i), nn.MaxPool2d(kernel_size=2,
                        stride=2, padding=0, ceil_mode=False))
            elif name == 'pool_ave':
                setattr(self, '_'+name+'_'+str(i), nn.AvgPool2d(kernel_size=2,
                        stride=2, padding=0, ceil_mode=False))
            elif name == 'concat':
                setattr(self, '_'+name+'_'+str(i), MyCat(dim=1))
            elif name == 'sum':  # ダミー
                setattr(self, '_'+name+'_'+str(i), MyCat(dim=1))
            elif "ConvBlock" in name:
                setattr(self, name+'_'+str(i), ConvBlock(kernel_size=int(name.split("_")[2]),
                        out_channels=int(name.split("_")[1]), device=device, rate_dropout=rate_dropout))
            elif "ResBlock" in name:
                setattr(self, name+'_'+str(i),
                        ResBlock(kernel_size=int(name.split("_")[2]),
                        out_channels=int(name.split("_")[1]), rate_dropout=rate_dropout,
                        device=device))
            elif "full" in name:  # 出力層
                setattr(self, name+'_'+str(i),
                        MyFullyConnectedBlock(out_features=n_class, device=device))
            elif "input" in name:
                pass
            else:
                raise ValueError(
                    f"not used function-name, name={name}")
            i += 1

        self.outputs = [None for _ in range(len(self.cgp))]
        self.param_num = 0

    def __call__(self, x):
        if self.search_space_obj.input_num > 1:
            self.outputs[:self.search_space_obj.input_num] = x
        else:
            self.outputs[0] = x  # 入力画像
        nodeID = self.search_space_obj.input_num

        param_num = 0

        for name, f in self.named_children():
            if 'ConvBlock' in name:
                self.outputs[nodeID], tmp_num = getattr(self, name)(
                    self.outputs[self.cgp[nodeID][1]])
                param_num += tmp_num
            elif 'ResBlock' in name:
                self.outputs[nodeID], tmp_num = getattr(self, name)(
                    self.outputs[self.cgp[nodeID][1]], self.outputs[self.cgp[nodeID][1]])
                param_num += tmp_num
            elif 'pool' in name:
                if self.outputs[self.cgp[nodeID][1]].shape[2] > 1:  # 画像サイズが 2 * 2 以上
                    self.outputs[nodeID] = f(
                        self.outputs[self.cgp[nodeID][1]])
                else:
                    self.outputs[nodeID] = self.outputs[self.cgp[nodeID][1]]
            elif 'concat' in name:
                in_data = [self.outputs[self.cgp[nodeID][1]],
                           self.outputs[self.cgp[nodeID][2]]]
                # 画像サイズの調整
                in_data = adjust_image_map_size(in_data=in_data)
                self.outputs[nodeID] = f((in_data[0], in_data[1]))
            elif 'sum' in name:
                in_data = [self.outputs[self.cgp[nodeID][1]],
                           self.outputs[self.cgp[nodeID][2]]]
                # 画像サイズの調整
                in_data = adjust_image_map_size(in_data=in_data)
                # チャネルサイズの調整
                in_data = adjust_image_channel_size(
                    in_data=in_data, device=self.device)
                self.outputs[nodeID] = in_data[0] + in_data[1]
            elif 'full' in name:
                in_data = [self.outputs[self.cgp[nodeID][j+1]]
                           for j in range(self.search_space_obj.out_type_in_num["full_"+name.split("_")[1]])]
                self.outputs[nodeID] = getattr(self, name)(in_data)
                param_num += f.linear.weight.shape[0] * \
                    f.linear.weight.shape[1] + f.linear.bias.shape[0]
            else:
                raise ValueError(
                    f"not defined function at CGPToCNN __call__ name={name}")
            nodeID += 1
        self.param_num = param_num

        return self.outputs[-1]
