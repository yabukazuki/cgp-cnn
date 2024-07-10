#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multiprocessing as mp
import numpy as np

import modules.cnn_train as cnn

# wrapper function for multiprocessing
def arg_wrapper_mp(args):
    return args[0](*args[1:])


# Evaluation of CNNs
def cnn_eval(net, gpu_id, epoch_num, batchsize, dataset, valid_data_ratio, verbose):

    print('\tgpu_id:', gpu_id, ',', net)
    train = cnn.CNN_train(dataset, is_valid=True,
                          valid_data_ratio=valid_data_ratio, verbose=verbose, search_space_obj=CgpInfoConvSet())
    evaluation = train(net, gpu_id, epoch_num=epoch_num, batchsize=batchsize, weight_decay=5e-4, eval_epoch_num=10,
                       data_aug=True, out_model=None, init_model=None, retrain_mode=False)
    print('\tgpu_id:', gpu_id, ', eval:', evaluation)
    return evaluation


class CNNEvaluation(object):
    def __init__(self, gpu_num, epoch_num, batchsize, dataset, valid_data_ratio, verbose):
        self.gpu_num = gpu_num
        self.epoch_num = epoch_num
        self.batchsize = batchsize
        self.dataset = dataset
        self.valid_data_ratio = valid_data_ratio
        self.verbose = verbose

    def __call__(self, net_lists):
        evaluations = np.zeros(len(net_lists))

        for i in np.arange(0, len(net_lists), self.gpu_num):
            process_num = np.min((i + self.gpu_num, len(net_lists))) - i

            pool = mp.Pool(process_num)
            arg_data = [(cnn_eval, net_lists[i+j], j, self.epoch_num, self.batchsize, self.dataset,
                         self.valid_data_ratio, self.verbose) for j in range(process_num)]
            evaluations[i:i+process_num] = pool.map(arg_wrapper_mp, arg_data)
            pool.terminate()

        return evaluations


class CgpInfoConvSet(object):
    def __init__(self, rows=5, cols=30, level_back=10, min_active_num=10, max_active_num=50):
        # network configurations depending on the problem
        self.input_num = 1

        self.func_type = ['ConvBlock_32_3', 'ConvBlock_32_5',
                          'ConvBlock_64_3', 'ConvBlock_64_5',
                          'ConvBlock_128_3', 'ConvBlock_128_5',
                          'pool_max', 'pool_ave',
                          'concat', 'sum']
        self.func_in_num = [1, 1,
                            1, 1,
                            1, 1,
                            1, 1,
                            2, 2]
        
        self.func_type_in_num = {}
        for func_type, func_in_num in zip(self.func_type, self.func_in_num):
            self.func_type_in_num[func_type] = func_in_num

        self.out_num = 1
        self.out_type = ['full_0']
        self.out_in_num = [1]
        
        self.out_type_in_num = {}
        for out_type, out_in_num in zip(self.out_type, self.out_in_num):
            self.out_type_in_num[out_type] = out_in_num

        # CGP network configuration
        self.rows = rows
        self.cols = cols
        self.node_num = rows * cols
        self.level_back = level_back
        self.min_active_num = min_active_num
        self.max_active_num = max_active_num

        self.func_type_num = len(self.func_type)
        self.out_type_num = len(self.out_type)
        self.max_in_num = np.max(
            [np.max(self.func_in_num), np.max(self.out_in_num)])


class CgpInfoResSet(object):
    def __init__(self, rows=5, cols=30, level_back=10, min_active_num=10, max_active_num=50):
        # network configurations depending on the problem
        self.input_num = 1

        self.func_type = ['ResBlock_32_3', 'ResBlock_32_5',
                          'ResBlock_64_3', 'ResBlock_64_5',
                          'ResBlock_128_3', 'ResBlock_128_5',
                          'pool_max', 'pool_ave',
                          'concat', 'sum']
        self.func_in_num = [1, 1,
                            1, 1,
                            1, 1,
                            1, 1,
                            2, 2]
        
        self.func_type_in_num = {}
        for func_type, func_in_num in zip(self.func_type, self.func_in_num):
            self.func_type_in_num[func_type] = func_in_num

        self.out_num = 1
        self.out_type = ['full_0']
        self.out_in_num = [1]
        
        self.out_type_in_num = {}
        for out_type, out_in_num in zip(self.out_type, self.out_in_num):
            self.out_type_in_num[out_type] = out_in_num

        # CGP network configuration
        self.rows = rows
        self.cols = cols
        self.node_num = rows * cols
        self.level_back = level_back
        self.min_active_num = min_active_num
        self.max_active_num = max_active_num

        self.func_type_num = len(self.func_type)
        self.out_type_num = len(self.out_type)
        self.max_in_num = np.max(
            [np.max(self.func_in_num), np.max(self.out_in_num)])


