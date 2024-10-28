#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
import graphviz

def make_architect_figure(net_list_source: list, out_file: str, search_space_obj):
    """個体のアーキテクチャ図の作成

    Args:
        net_list_source (list): 個体の遺伝子 (net_list), ただし層の種類を名前に置換したもの
        out_file (str): 出力ファイル名

    """
    out_file += ".dot"
    net_list = deepcopy(net_list_source)
    dot = graphviz.Digraph(format='jpeg')
    dot.attr('node', fontname='Meiryo UI')
    dot.attr('edge', fontname='Meiryo UI')

    for i, layer in enumerate(net_list):
        if "skipconcatAuxiliary" in layer[0]:
            dot.node(layer[0]+"__"+str(i), layer[0], shape="box")
            dot.node(layer[0]+"__"+str(i)+"_auxiliary_output",
                     "auxiliary_output_"+str(i), shape="box")
            net_list[i][0] += "__"+str(i)
        elif "skipconcatNonAuxiliary" in layer[0]:
            dot.node(layer[0]+"__"+str(i), layer[0], shape="box")
            net_list[i][0] += "__"+str(i)
        else:
            dot.node(layer[0]+"__"+str(i), layer[0], shape="box")
            net_list[i][0] += "__"+str(i)

    for layer in net_list[search_space_obj.input_num:-search_space_obj.out_num]:
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
