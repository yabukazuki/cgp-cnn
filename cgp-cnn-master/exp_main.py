#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("./tools")
from tools.dataset_class import DataSet

import argparse
import pickle
import pandas as pd

from modules.cgp import *
from modules.cgp_config import *
from modules.cnn_train import CNN_train


if __name__ == '__main__':

    func_set = {
        'ConvSet': CgpInfoConvSet,
        'ResSet': CgpInfoResSet,
    }

    parser = argparse.ArgumentParser(description='Evolving CNN structures of GECCO 2017 paper')
    parser.add_argument('--func_set', '-f', choices=func_set.keys(), default='ConvSet', help='Function set of CGP (ConvSet or ResSet)')
    parser.add_argument('--gpu_num', '-g', type=int, default=1, help='Num. of GPUs')
    parser.add_argument('--lam', '-l', type=int, default=2, help='Num. of offsprings')
    parser.add_argument('--net_info_file', default='network_info.pickle', help='Network information file name')
    parser.add_argument('--log_file', default='./log_cgp.txt', help='Log file name')
    parser.add_argument('--mode', '-m', default='evolution', help='Mode (evolution / retrain)')
    args = parser.parse_args()

    # --- Optimization of the CNN architecture ---
    if args.mode == 'evolution':
        # Create CGP configuration and save network information
        network_info = func_set[args.func_set](rows=5, cols=30, level_back=10, min_active_num=10, max_active_num=50)
        with open(args.net_info_file, mode='wb') as f:
            pickle.dump(network_info, f)

        # Evaluation function for CGP (training CNN and return validation accuracy)
        eval_f = CNNEvaluation(gpu_num=args.gpu_num, dataset='comic_one_input', valid_data_ratio=0.2, verbose=True,
                               epoch_num=50, batchsize=80)

        # Execute evolution
        cgp = CGP(network_info, eval_f, lam=args.lam)
        cgp.modified_evolution(max_eval=50000, mutation_rate=0.05, log_file=args.log_file)

    # --- Retraining evolved architecture ---
    elif args.mode == 'retrain':
        # Load CGP configuration
        with open(args.net_info_file, mode='rb') as f:
            network_info = pickle.load(f)

        # Load network architecture
        cgp = CGP(network_info, None)
        data = pd.read_csv(args.log_file, header=None)  # Load log file
        cgp.load_log(list(data.tail(1).values.flatten().astype(int)))  # Read the log at final generation

        # Retraining the network
        temp = CNN_train('comic_one_input', is_valid=False, valid_data_ratio=0.2, verbose=True, search_space_obj=func_set[args.func_set]())
        # TODO: パラメータを合わせる
        acc = temp(cgp.pop[0].active_net_list(), 0, epoch_num=100, batchsize=20, weight_decay=5e-4, eval_epoch_num=1,
                   data_aug=True, out_model='retrained_net.model', init_model=None, retrain_mode=True)
        print(acc)

    else:
        print('Undefined mode.')
