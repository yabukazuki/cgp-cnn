#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import time
import time
import numpy as np

import matplotlib.pyplot as plt

from modules.make_architect_figure import make_architect_figure


class Individual(object):

    def __init__(self, net_info):
        self.net_info = net_info
        self.gene = np.zeros(
            (self.net_info.node_num + self.net_info.out_num, self.net_info.max_in_num + 1)).astype(int)
        self.is_active = np.empty(
            self.net_info.node_num + self.net_info.out_num).astype(bool)
        self.eval = None
        self.init_gene()

    def init_gene(self):
        # intermediate node
        for n in range(self.net_info.node_num + self.net_info.out_num):
            # type gene
            type_num = self.net_info.func_type_num if n < self.net_info.node_num else self.net_info.out_type_num
            self.gene[n][0] = np.random.randint(type_num)
            # connection gene
            col = np.min((int(n / self.net_info.rows), self.net_info.cols))
            max_connect_id = col * self.net_info.rows + self.net_info.input_num
            min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
                if col - self.net_info.level_back >= 0 else 0
            for i in range(self.net_info.max_in_num):
                self.gene[n][i + 1] = min_connect_id + \
                    np.random.randint(max_connect_id - min_connect_id)

        self.check_active()

    def __check_course_to_out(self, n):
        if not self.is_active[n]:
            self.is_active[n] = True
            t = self.gene[n][0]
            if n >= self.net_info.node_num:    # output node
                in_num = self.net_info.out_in_num[t]
            else:    # intermediate node
                in_num = self.net_info.func_in_num[t]

            for i in range(in_num):
                if self.gene[n][i+1] >= self.net_info.input_num:
                    self.__check_course_to_out(
                        self.gene[n][i+1] - self.net_info.input_num)

    def check_active(self):
        # clear
        self.is_active[:] = False
        # start from output nodes
        for n in range(self.net_info.out_num):
            self.__check_course_to_out(self.net_info.node_num + n)

    def __mutate(self, current, min_int, max_int):
        mutated_gene = current
        while current == mutated_gene:
            mutated_gene = min_int + np.random.randint(max_int - min_int)
        return mutated_gene

    def mutation(self, mutation_rate):
        active_check = False

        for n in range(self.net_info.node_num + self.net_info.out_num):
            t = self.gene[n][0]
            # mutation for type gene
            type_num = self.net_info.func_type_num if n < self.net_info.node_num else self.net_info.out_type_num
            if np.random.rand() < mutation_rate and type_num > 1:
                self.gene[n][0] = self.__mutate(self.gene[n][0], 0, type_num)
                if self.is_active[n]:
                    active_check = True
            # mutation for connection gene
            col = np.min((int(n / self.net_info.rows), self.net_info.cols))
            max_connect_id = col * self.net_info.rows + self.net_info.input_num
            min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
                if col - self.net_info.level_back >= 0 else 0
            in_num = self.net_info.func_in_num[t] if n < self.net_info.node_num else self.net_info.out_in_num[t]
            for i in range(self.net_info.max_in_num):
                if np.random.rand() < mutation_rate and max_connect_id - min_connect_id > 1:
                    self.gene[n][i+1] = self.__mutate(
                        self.gene[n][i+1], min_connect_id, max_connect_id)
                    if self.is_active[n] and i < in_num:
                        active_check = True

        self.check_active()
        return active_check

    def neutral_mutation(self, mutation_rate):
        for n in range(self.net_info.node_num + self.net_info.out_num):
            t = self.gene[n][0]
            # mutation for type gene
            type_num = self.net_info.func_type_num if n < self.net_info.node_num else self.net_info.out_type_num
            if not self.is_active[n] and np.random.rand() < mutation_rate and type_num > 1:
                self.gene[n][0] = self.__mutate(self.gene[n][0], 0, type_num)
            # mutation for connection gene
            col = np.min((int(n / self.net_info.rows), self.net_info.cols))
            max_connect_id = col * self.net_info.rows + self.net_info.input_num
            min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
                if col - self.net_info.level_back >= 0 else 0
            in_num = self.net_info.func_in_num[t] if n < self.net_info.node_num else self.net_info.out_in_num[t]
            for i in range(self.net_info.max_in_num):
                if (not self.is_active[n] or i >= in_num) and np.random.rand() < mutation_rate \
                        and max_connect_id - min_connect_id > 1:
                    self.gene[n][i+1] = self.__mutate(
                        self.gene[n][i+1], min_connect_id, max_connect_id)

        self.check_active()
        return False

    def count_active_node(self):
        return self.is_active.sum()

    def copy(self, source):
        self.net_info = source.net_info
        self.gene = source.gene.copy()
        self.is_active = source.is_active.copy()
        self.eval = source.eval

    def active_net_list(self):
        net_list = [["input_0", 0, 0]]
        active_cnt = np.arange(self.net_info.input_num +
                               self.net_info.node_num + self.net_info.out_num)
        active_cnt[self.net_info.input_num:] = np.cumsum(self.is_active)

        for n, is_a in enumerate(self.is_active):
            if is_a:
                t = self.gene[n][0]
                if n < self.net_info.node_num:    # intermediate node
                    type_str = self.net_info.func_type[t]
                else:    # output node
                    type_str = self.net_info.out_type[t]

                connections = [active_cnt[self.gene[n][i+1]]
                               for i in range(self.net_info.max_in_num)]
                net_list.append([type_str] + connections)
        return net_list


# CGP with (1 + \lambda)-ES
class CGP(object):

    def __init__(self, net_info, eval_func, lam):
        self.lam = lam
        self.net_info = net_info
        self.pop = [Individual(self.net_info) for _ in range(1 + self.lam)]
        self.eval_func = eval_func

        self.num_gen = 0
        self.num_eval = 0

    def _evaluation(self, pop, eval_flag):
        # create network list
        net_lists = []
        active_index = np.where(eval_flag)[0]
        for i in active_index:
            net_lists.append(pop[i].active_net_list())

        # evaluation
        fp = self.eval_func(net_lists)
        for i, j in enumerate(active_index):
            pop[j].eval = fp[i]
        evaluations = np.zeros(len(pop))
        for i in range(len(pop)):
            evaluations[i] = pop[i].eval

        self.num_eval += len(net_lists)
        return evaluations

    def _log_data(self, net_info_type):
        log_list = [self.num_gen, self.num_eval, time.clock(
        ), self.pop[0].eval, self.pop[0].count_active_node()]
        if net_info_type == 'active_only':
            log_list.append(self.pop[0].active_net_list())
        elif net_info_type == 'full':
            log_list += self.pop[0].gene.flatten().tolist()
        else:
            pass
        return log_list

    def load_log(self, log_data):
        self.num_gen = log_data[0]
        self.num_eval = log_data[1]
        net_info = self.pop[0].net_info
        self.pop[0].eval = log_data[3]
        self.pop[0].gene = np.array(log_data[5:]).reshape(
            (net_info.node_num + net_info.out_num, net_info.max_in_num + 1))
        self.pop[0].check_active()

    # Usual evolution procedure of CGP (This is not used for GECCO 2017 paper)
    def evolution(self, max_eval, mutation_rate, log_file):
        with open(log_file, 'w') as fw:
            writer = csv.writer(fw, lineterminator='\n')

            eval_flag = np.empty(self.lam)

            self._evaluation([self.pop[0]], np.array([True]))
            print(self._log_data(net_info_type='active_only'))

            while self.num_eval < max_eval:
                self.num_gen += 1

                # reproduction
                for i in range(self.lam):
                    self.pop[i+1].copy(self.pop[0])    # copy a parent
                    eval_flag[i] = self.pop[i +
                                            # mutation
                                            1].mutation(mutation_rate)

                # evaluation and selection
                evaluations = self._evaluation(
                    self.pop[1:], eval_flag=eval_flag)
                best_arg = evaluations.argmax()
                if evaluations[best_arg] >= self.pop[0].eval:
                    self.pop[0].copy(self.pop[best_arg+1])

                # display and save log
                if eval_flag.sum() > 0:
                    print(self._log_data(net_info_type='active_only'))
                    writer.writerow(self._log_data(net_info_type='full'))

    # Modified CGP (used for GECCO 2017 paper):
    #   At each iteration:
    #     - Generate lambda individuals in which at least one active node changes (i.e., forced mutation)
    #     - Mutate the best individual with neutral mutation (unchanging the active nodes)
    #         if the best individual is not updated.
    def modified_evolution(self, max_eval, mutation_rate, log_file):
        start = time.time()
        with open(log_file, 'w') as fw:
            writer = csv.writer(fw, lineterminator='\n')

            eval_flag = np.empty(self.lam)

            active_num = self.pop[0].count_active_node()
            while active_num < self.pop[0].net_info.min_active_num or active_num > self.pop[0].net_info.max_active_num:
                self.pop[0].mutation(1.0)
                active_num = self.pop[0].count_active_node()
            self._evaluation([self.pop[0]], np.array([True]))
            print(self._log_data(net_info_type='active_only'))

            fitnesses_gen = [self.pop[0].eval]
            while self.num_eval < max_eval and self.num_gen < 200:
                self.num_gen += 1

                # reproduction
                for i in range(self.lam):
                    eval_flag[i] = False
                    self.pop[i + 1].copy(self.pop[0])  # copy a parent
                    active_num = self.pop[i + 1].count_active_node()

                    # forced mutation
                    while not eval_flag[i] or active_num < self.pop[i + 1].net_info.min_active_num \
                            or active_num > self.pop[i + 1].net_info.max_active_num:
                        self.pop[i + 1].copy(self.pop[0])  # copy a parent
                        eval_flag[i] = self.pop[i +
                                                # mutation
                                                1].mutation(mutation_rate)
                        active_num = self.pop[i + 1].count_active_node()

                # evaluation and selection
                evaluations = self._evaluation(
                    self.pop[1:], eval_flag=eval_flag)
                best_arg = evaluations.argmax()
                if evaluations[best_arg] > self.pop[0].eval:
                    self.pop[0].copy(self.pop[best_arg + 1])
                else:
                    self.pop[0].neutral_mutation(
                        mutation_rate)  # neutral mutation

                print()
                print(self.pop[0].active_net_list())
                print(f"fitness = {self.pop[0].eval}")
                print(f"num_gen = {self.num_gen}")
                print(f"num_eval = {self.num_eval}")
                fitnesses_gen.append(self.pop[0].eval)

                plt.figure()
                plt.plot(range(len(fitnesses_gen)),
                         fitnesses_gen, color="blue")
                plt.xlabel("Generation")
                plt.ylabel("Fitness")
                plt.grid(True)
                # plt.legend(bbox_to_anchor=(1.05, 1),
                #         loc='upper left', borderaxespad=0)
                plt.savefig(f"fitnesses_gene.eps", bbox_inches="tight")
                plt.close()

                with open(f"csv.result.d/fitnesses_gene.csv", 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(list(range(len(fitnesses_gen))))
                    writer.writerow(fitnesses_gen)

                with open("experiment_memo.txt", "a") as f:
                    f.write(f"fitness = {self.pop[0].eval}\n")
                    f.write(f"num_gen = {self.num_gen}\n")
                    f.write(f"num_eval = {self.num_eval}\n")
                    f.write(f"time = {(time.time()-start)/60.0} minutes\n\n")
                print(f"time = {(time.time()-start)/60.0} minutes")
                print()
                make_architect_figure(net_list_source=self.pop[0].active_net_list(
                ), out_file=f"best_indiv", search_space_obj=self.net_info)
            return self.pop[0]
