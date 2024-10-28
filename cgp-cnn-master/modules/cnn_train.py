#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import random
import csv

from modules.cnn_model import CGP2CNN
from modules.cgp_config import *

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets
from torch.optim.lr_scheduler import CosineAnnealingLR

from modules.extend_dataset import apply_augmentation_with_original32
from modules.extend_dataset import apply_augmentation_with_original32_test
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# __init__: load dataset
# __call__: training the CNN defined by CGP list

class CgpInfoConvSet(object):
    def __init__(self, rows=5, cols=30, level_back=10, min_active_num=10, max_active_num=50):
        # network configurations depending on the problem
        self.input_num = 1

        self.func_type = ['ConvBlock_32_1', 'ConvBlock_32_3', 'ConvBlock_32_5',
                          'ConvBlock_64_1', 'ConvBlock_64_3', 'ConvBlock_64_5',
                          'ConvBlock_128_1', 'ConvBlock_128_3', 'ConvBlock_128_5',
                          'pool_max', 'pool_ave',
                          'sum', 'concat']
        self.func_in_num = [1, 1, 1,
                            1, 1, 1,
                            1, 1, 1,
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


class CNN_train():
    def __init__(self, dataset_name, is_valid, valid_data_ratio, verbose, search_space_obj):
        # dataset_name: name of data set ('cifar10' or 'cifar100' or 'mnist')
        # is_valid: [True]  model validation mode
        #                     (split training data set according to valid_data_ratio for evaluation of CGP individual)
        #             [False] model test mode for final evaluation of the evolved model
        #                     (raining data : all training data, test data : all test data)
        # valid_data_ratio: ratio of the validation data
        #                    (e.g., if the number of all training data=50000 and valid_data_ratio=0.2,
        #                       the number of training data=40000, validation=10000)
        # verbose: flag of display
        self.verbose = verbose
        self.search_space_obj = search_space_obj

        if "comic" in dataset_name:
            if dataset_name == "comic_one_input":
                self.n_class = 5
                self.channel = 1
                self.pad_size = 4

                root = "./data"
                filename_train = root + "/4koma_comic/train"
                filename_test = root + "/4koma_comic/test"
                with open(filename_train, mode="rb") as f_train:
                    train = pickle.load(f_train)
                with open(filename_test, mode="rb") as f_test:
                    test = pickle.load(f_test)
                if is_valid:
                    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(train.data, train.targets, test_size=valid_data_ratio, random_state=0,
                                                                                        shuffle=True, stratify=train.targets)
                    self.x_train, self.y_train = apply_augmentation_with_original32(
                        self.x_train, self.y_train, num_times=4)
                    self.x_test = torch.tensor(apply_augmentation_with_original32_test(self.x_test).transpose(
                        0, 3, 1, 2), dtype=torch.float32)
                    # 次元移動
                    # tensor にする
                    self.x_train = torch.tensor(self.x_train.transpose(
                        0, 3, 1, 2), dtype=torch.float32)
                    self.y_train = torch.tensor(
                        self.y_train, dtype=torch.long)

                    self.y_test = torch.tensor(
                        self.y_test, dtype=torch.long)
                else:
                    self.x_train, _, self.y_train, _ = train_test_split(torch.tensor(train.data.transpose(0, 3, 1, 2)/255.0, dtype=torch.float32), torch.tensor(train.targets, dtype=torch.long), test_size=valid_data_ratio, random_state=0,
                                                                        shuffle=True, stratify=train.targets)
                    self.x_test, self.y_test = torch.tensor(test.data.transpose(
                        0, 3, 1, 2)/255.0, dtype=torch.float32), torch.tensor(test.targets, dtype=torch.long)

            elif dataset_name == "comic_two_inputs":
                self.n_class = 2
                self.channel = 3
                self.pad_size = 4

                root = "./data"
                filename_train = root + "/4koma_comic/train"
                filename_test = root + "/4koma_comic/test"
                with open(filename_train, mode="rb") as f_train:
                    train = pickle.load(f_train)
                with open(filename_test, mode="rb") as f_test:
                    test = pickle.load(f_test)

                if is_valid:
                    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(torch.tensor(train.data.transpose(0, 1, 4, 2, 3)/255.0, dtype=torch.float32), torch.tensor(train.targets, dtype=torch.long), test_size=valid_data_ratio, random_state=0,
                                                                                            shuffle=True, stratify=train.targets)
                else:
                    self.x_train, _, self.y_train, _ = train_test_split(torch.tensor(train.data.transpose(0, 1, 4, 2, 3)/255.0, dtype=torch.float32), torch.tensor(train.targets, dtype=torch.long), test_size=valid_data_ratio, random_state=0,
                                                                        shuffle=True, stratify=train.targets)
                    self.x_test, self.y_test = torch.tensor(test.data.transpose(
                        0, 1, 4, 2, 3)/255.0, dtype=torch.float32), torch.tensor(test.targets, dtype=torch.long)
            else:
                raise ValueError(
                    f"invalid dataset name dataset = {dataset_name}")

        else:
            download = True
            if dataset_name == 'cifar10':
                self.n_class = 10
                self.channel = 3
                self.pad_size = 4
                train, test = datasets.CIFAR10(
                    root='../data', train=True, download=download), datasets.CIFAR10(
                    root='../data', train=False, download=download)
            elif dataset_name == 'cifar100':
                self.n_class = 100
                self.channel = 3
                self.pad_size = 4
                train, test = datasets.CIFAR100(
                    root='../data', train=True, download=download), datasets.CIFAR100(
                    root='../data', train=False, download=download)
            elif dataset_name == "mnist":
                self.n_class = 10
                self.channel = 1
                self.pad_size = 4
                train, test = datasets.MNIST(
                    root='../data', train=True, download=download), datasets.MNIST(
                    root='../data', train=False, download=download)

                # mnist のようにデータが 2 次元配列の場合に 3 次元配列に変換
                train.data = train.data.view(len(train.data), len(
                    train.data[0]), len(train.data[0]), 1)
                train.data = train.data.to("cpu").detach(
                ).numpy().copy()

                test.data = test.data.view(len(test.data), len(
                    test.data[0]), len(test.data[0]), 1)
                test.data = test.data.to("cpu").detach(
                ).numpy().copy()

            else:
                raise ValueError(
                    f"not defined dataset, dataset={dataset_name}")

            if is_valid:
                self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(torch.tensor(train.data.transpose(0, 3, 1, 2)/255.0, dtype=torch.float32), torch.tensor(train.targets, dtype=torch.long), test_size=valid_data_ratio, random_state=0,
                                                                                        shuffle=True, stratify=train.targets)
            else:
                self.x_train, _, self.y_train, _ = train_test_split(torch.tensor(train.data.transpose(0, 3, 1, 2)/255.0, dtype=torch.float32), torch.tensor(train.targets, dtype=torch.long), test_size=valid_data_ratio, random_state=0,
                                                                    shuffle=True, stratify=train.targets)
                self.x_test, self.y_test = torch.tensor(test.data.transpose(
                    0, 3, 1, 2)/255.0, dtype=torch.float32), torch.tensor(test.targets, dtype=torch.long)

        # 標準化
        if self.channel == 1:
            image_list = []
            for x in self.x_train:
                image_list.append(x)
            all_images = torch.stack(image_list)
            mean = all_images.mean()
            std = all_images.std()
            self.x_train = (self.x_train - mean) / std
            self.x_test = (self.x_test - mean) / std
            print(f'Mean: {mean.item()}, Std: {std.item()}')
        
        self.train_data_num = len(self.x_train)
        self.test_data_num = len(self.x_test)
        if self.verbose:
            print('\ttrain data shape:', self.x_train.shape)
            print('\ttest data shape :', self.x_test.shape)

    def __call__(self, cgp, gpuID, epoch_num, batchsize, eval_epoch_num, weight_decay,
                 data_aug, out_model, init_model,
                 retrain_mode):
        if self.verbose:
            print('\tGPUID    :', gpuID)
            print('\tepoch_num:', epoch_num)
            print('\tbatchsize:', batchsize)

        device = "cuda:" + str(gpuID)  # Make a specified GPU current

        if init_model is not None:
            if self.verbose:
                print('\tLoad model from', init_model)
            model = torch.load(init_model)
        else:
            search_space_obj=CgpInfoConvSet()
            model = CGP2CNN(cgp, self.n_class, device, rate_dropout=0.3, search_space_obj=search_space_obj)
        model.to(device)

        # モデルの点検 (例) と各層への入力チャネルや次元のサイズ設定
        x_check = self.x_train[:2]
        if self.search_space_obj.input_num > 1:
            x_check = torch.transpose(x_check, 1, 0)
        model.eval()
        x_check = x_check.to(device)
        try:
            with torch.no_grad():
                model(x_check)
        except:
            import traceback
            traceback.print_exc()
            return 0.

        model.train()

        criterion = nn.CrossEntropyLoss()
        # TODO: テスト実験における学習率の最適化
        lr = 0.005 if not retrain_mode else 0.005
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08,
                                weight_decay=weight_decay)
        # scheduler = StepLR(
        #         optimizer, step_size=self.crossover_epoch, gamma=0.80, verbose=True)

        train_accuracies, train_losses = np.zeros(
            epoch_num), np.zeros(epoch_num)
        test_accuracies, test_losses = np.zeros(
            epoch_num), np.zeros(epoch_num)

        fix_random_seed(0)  # 個体ごとに 1 epoch 目に訓練に用いるデータの順番 (perm) を同じにするため

        for epoch in range(1, epoch_num+1, 1):
            model.train()
            if self.verbose:
                print('\tepoch', epoch)
            perm = np.random.permutation(self.train_data_num)
            train_accuracy = train_loss = 0
            start = time.time()
            for i in range(0, self.train_data_num, batchsize):
                xx_train = self.x_train[perm[i:i + batchsize]]
                if self.search_space_obj.input_num > 1:
                    xx_train = torch.transpose(xx_train, 1, 0)
                x = xx_train.to(device)
                t = self.y_train[perm[i:i + batchsize]].to(device)

                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, t)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                _, predict = torch.max(outputs.data, 1)
                train_accuracy += float((predict == t).sum().item())

            # if retrain_mode:
            #     scheduler.step()

            elapsed_time = time.time() - start
            throughput = self.train_data_num / elapsed_time

            train_losses[epoch - 1] = train_loss / self.train_data_num
            train_accuracies[epoch - 1] = train_accuracy / self.train_data_num

            test_accuracy, test_loss = self.__test(
                model, batchsize=512, device=device)
            test_losses[epoch - 1] = test_loss / self.test_data_num
            test_accuracies[epoch - 1] = test_accuracy / self.test_data_num

            if self.verbose:
                print('\ttrain mean loss={}, train accuracy={}, time={}, throughput={} images/sec, paramNum={}'.format(
                    train_losses[epoch - 1], train_accuracies[epoch - 1], elapsed_time, throughput, model.param_num))
                print('\ttest mean loss={}, test accuracy={}, time={}, throughput={} images/sec, paramNum={}'.format(
                    test_losses[epoch - 1], test_accuracies[epoch - 1], elapsed_time, throughput, model.param_num))

        # test_accuracy, test_loss = self.__test(model, batchsize=512, device=device)
        if out_model is not None:
            model.to("cpu")
            torch.save(model, out_model)
        if retrain_mode:
            CSV_SAVE_DIR = "csv.result.d"
            # ロスの推移
            plt.figure()
            plt.plot(range(1, len(train_losses)+1, 1),
                     train_losses, label="Train loss", color="blue")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.05, 1),
                       loc='upper left', borderaxespad=0)
            plt.savefig(f"loss_{out_model}.eps", bbox_inches="tight")
            plt.close()

            with open(f"{CSV_SAVE_DIR}/loss_{out_model}.csv", 'a') as f:
                writer = csv.writer(f)
                writer.writerow(list(range(1, len(train_losses)+1, 1)))
                writer.writerow(train_losses)

            # 正答率の推移
            plt.figure()
            plt.plot(range(1, len(train_accuracies)+1, 1),
                     train_accuracies, label="Train accuracy", color="blue")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.ylim(-0.05, 1.05)
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.05, 1),
                       loc='upper left', borderaxespad=0)
            plt.savefig(
                f"accuracy_{out_model}.eps", bbox_inches="tight")
            plt.close()

            with open(f'{CSV_SAVE_DIR}/accuracy_{out_model}.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(list(range(1, len(train_accuracies)+1, 1)))
                writer.writerow(train_accuracies)

        return np.max(test_accuracies[-eval_epoch_num:])

    def test(self, cgp, model_file, batchsize):

        device = "cuda:0"  # Make a specified GPU current
        model = CGP2CNN(cgp, self.n_class)
        print('\tLoad model from', model_file)
        model = torch.load(model_file)
        model.to(device)

        model.eval()
        criterion = nn.CrossEntropyLoss()
        test_accuracy = test_loss = 0.0

        # CSVファイルのヘッダー
        csv_header = ['data', 'true_label',
                      'predicted_label', 'correct_prediction']
        # CSVファイルを開き、結果を書き込む
        with open('test_results.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(csv_header)  # ヘッダーを書き込む
        all_predictions = []
        all_targets = []

        f = open('test_results.csv', mode='a', newline='')
        writer = csv.writer(f)

        for i in range(0, self.test_data_num, batchsize):
            x = self.x_test[i:i + batchsize]
            if self.search_space_obj.input_num > 1:
                x = torch.transpose(x, 1, 0)
            x = x.to(device)
            t = self.y_test[i:i + batchsize].to(device)
            with torch.no_grad():
                outputs = model(x)
                loss = criterion(outputs, t)
                _, predict = torch.max(outputs.data, 1)
            test_loss += loss.item()
            test_accuracy += float((predict == t).sum().item())

            all_predictions.extend(predict.cpu().numpy())
            all_targets.extend(t.cpu().numpy())

            # バッチごとのデータをCSVに書き込む
            for j in range(batchsize):
                # データポイントがバッチの最後でないことを確認
                if i + j < self.test_data_num:
                    data_point = x[j].cpu().numpy()  # データポイント
                    true_label = t[j].item()  # 正しいラベル
                    predicted_label = predict[j].item()  # 予測したラベル
                    correct = 1 if true_label == predicted_label else 0  # 正解フラグ

                    # CSVに書き込む行の作成
                    row = [i + j + 1, true_label, predicted_label, correct]
                    writer.writerow(row)

        print('\tparamNum={}'.format(model.param_num))
        print('\ttest mean loss={}, test accuracy={}'.format(
            test_loss / self.test_data_num, test_accuracy / self.test_data_num))

        cm = confusion_matrix(all_targets, all_predictions)
        print(f"混同行列:\n{cm}")

        correct_predictions = np.diag(cm)
        for i, correct_count in enumerate(correct_predictions):
            print(f'クラス {i} の正解数: {correct_count}')
        f.close()

        # 混同行列を正規化
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # 温度マップとして混同行列を描画
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        # 画像として保存
        plt.savefig('confusion_matrix_heatmap.png')

        return test_accuracy, test_loss

    def __test(self, model, batchsize, device):
        model.eval()
        criterion = nn.CrossEntropyLoss()
        test_accuracy = test_loss = 0.0
        for i in range(0, self.test_data_num, batchsize):
            x = self.x_test[i:i + batchsize]
            if self.search_space_obj.input_num > 1:
                x = torch.transpose(x, 1, 0)
            x = x.to(device)
            t = self.y_test[i:i + batchsize].to(device)
            with torch.no_grad():
                outputs = model(x)
                loss = criterion(outputs, t)
                _, predict = torch.max(outputs.data, 1)
            test_loss += loss.item()
            test_accuracy += float((predict == t).sum().item())
        return test_accuracy, test_loss

