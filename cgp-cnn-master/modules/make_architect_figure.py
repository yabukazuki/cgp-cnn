#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
import graphviz

def make_architect_figure(net_list_source: list, out_file: str, search_space_obj):
    """個体のアーキテクチャ図の作成とファイルへの保存

    Args:
        net_list_source (list): 個体の遺伝子リスト (層の種類を名前に置換したもの)。
        out_file (str): 出力ファイル名 (拡張子を含まない)。
        search_space_obj (SearchSpace): 入力と出力の層数や各層の種類や入力数を定義するオブジェクト。
    """
    out_file += ".dot"
    net_list = deepcopy(net_list_source)
    dot = graphviz.Digraph(format='jpg')
    dot.attr('node', fontname='Meiryo UI')
    dot.attr('edge', fontname='Meiryo UI')
    dot.attr(rankdir='BT')

    # 層の種類に応じた色の定義
    color_map = {
        "ConvBlock": "lightblue",
        "pool": "pink",
        "sum": "green",
        "concat": "lightyellow",
        "skipconcatAuxiliary": "white",
        "skipconcatNonAuxiliary": "white"
    }

    for i, layer in enumerate(net_list):
        layer_name = layer[0] + "__" + str(i)
        shape = "box"
        color = color_map.get(layer[0].split("_")[0], "white")  # デフォルトは白

        # skipconcatAuxiliary タイプのノード作成
        if "skipconcatAuxiliary" in layer[0]:
            label = "Concatenation\n crossover point"
            dot.node(layer_name+"_auxiliary_output",
                    "Auxiliary output", shape=shape, style="filled", fillcolor=color, color="black")
        else:
            if layer[0].startswith("ConvBlock"):
                out_channels = layer[0].split('_')[1]
                kernel_size = f"{layer[0].split('_')[2]} ✕ {layer[0].split('_')[2]}"
                # label = f"{{ ReLU | Batch Norm | Dropout (0.3) | Convolution Block ({kernel_size}, {out_channels}) }}"
                label = f"Convolution Block\n kernel size: {kernel_size}\n Out ch: {out_channels}"
            elif layer[0].startswith("pool"):
                label = f"Pooling\n {layer[0].split('_')[1]}"
            elif layer[0].startswith("sum"):
                label = "Summation"
            elif layer[0].startswith("concat"):
                label = "Concatenation"
            elif layer[0].startswith("input"):
                label = "Input"
            elif layer[0].startswith("skipconcatNonAuxiliary"):
                label = "Concatenation\n crossover point"
            elif layer[0].startswith("full"):
                label = "Output"
            else:
                label = layer[0]
        dot.node(layer_name, label=label, shape=shape, style="filled", fillcolor=color, color="black")

        net_list[i][0] += "__" + str(i)

    # 中間層のエッジ作成
    for idx, layer in enumerate(net_list[search_space_obj.input_num:-search_space_obj.out_num]):
        if "skipconcatAuxiliary" in layer[0]:
            for i in range(2):
                dot.edge(net_list[layer[i+1]][0], layer[0], label="")
            dot.edge(net_list[layer[2]][0], layer[0] +
                     "_auxiliary_output", label="")
        elif "skipconcatNonAuxiliary" in layer[0]:
            for i in range(2):
                dot.edge(net_list[layer[i+1]][0], layer[0], label="")
        else:
            for i in range(search_space_obj.func_type_in_num[layer[0].split("__")[0]]):
                dot.edge(net_list[layer[i+1]][0], layer[0], label="")
    for layer in net_list[-search_space_obj.out_num:]:
        for i in range(search_space_obj.out_type_in_num[layer[0].split("__")[0]]):
            dot.edge(net_list[layer[i+1]][0], layer[0], label="")

    dot.render(out_file, view=False)
