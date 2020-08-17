import copy
import torch
import numpy as np
import json
import os

from coding import Coding
from requirement_reader import read_all_labeled_stories

START_TAG = "<START>"
STOP_TAG = "<STOP>"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DataManager:
    def __init__(self,
                 training_id,
                 is_pretraining=True,
                 batch_size=20,
                 pretraining_batch_size=20,
                 experiment_option='evaluation',
                 coding_level='char'):
        self.training_id = training_id
        self.index = 0
        self.input_size = 0
        self.batch_size = batch_size
        self.pretraining_batch_size = pretraining_batch_size

        self.is_pretraining = is_pretraining
        self.experiment_option = experiment_option

        self.batch_training_data = []
        self.batch_testing_data = []
        self.batch_pretraining_data = []
        self.batch_tuning_data = []

        # 判断实验设置选项
        # evaluation：进行实验评估
        # traning: 所有标注数据都拿来训练
        assert experiment_option in ['evaluation', 'training']

        self.training_data, self.testing_data = self.load_data()
        if self.is_pretraining:
            self.pretraining_data = self.load_pretraining_data()
            self.tuning_data = self.load_tuning_data()

        self.vocabulary = {'unk': 0}
        self.tag_map = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
        self.build_vocabulary()

        self.training_coding = Coding(vocabulary=self.vocabulary,
                                      coding_level=coding_level,
                                      original_pairs=self.training_data)
        self.testing_coding = None if len(self.testing_data) == 0 else \
            Coding(vocabulary=self.vocabulary,
                   coding_level=coding_level, original_pairs=self.testing_data)
        if self.is_pretraining:
            self.pretraining_coding = Coding(vocabulary=self.vocabulary, coding_level=coding_level,
                                             original_pairs=self.pretraining_data, is_pretraining=True)
            self.pretraining_coding = Coding(vocabulary=self.vocabulary, coding_level=coding_level,
                                             original_pairs=self.pretraining_data, is_pretraining=True)

        assert (experiment_option == 'training' and self.testing_coding is None) or \
               (experiment_option == 'evaluation' and self.testing_coding is not None)

        self.max_length = max(self.training_coding.max_input_length, self.testing_coding.max_input_length) \
                        if self.testing_coding is not None else self.training_coding.max_input_length
        print('BATCHING')
        self.prepare_batch()
        print('BATCHING FINISHED!!!')

    def load_data(self):
        """
        加载数据
        :return:
        """
        original_pairs = read_all_labeled_stories()

        # training mode
        if self.experiment_option == 'training':
            return original_pairs, []

        # evaluation mode
        np.random.shuffle(original_pairs)
        split_index = int(len(original_pairs)/10)*9
        return original_pairs[0: split_index], original_pairs[split_index:]

    def load_pretraining_data(self):
        """
        读取预训练所有数据
        :return:
        """
        sentences = []
        with open('wiki_data/pretraining_corpus.txt', 'r', encoding='utf-8') as file:
            for line in file.readlines():
                line = line.strip('\n')
                if len(line) == 0:
                    continue
                sentences.append((line, []))
        np.random.shuffle(sentences)
        return sentences

    def load_tuning_data(self):
        """
        读取预训练所有数据
        :return:
        """
        sentences = []
        with open('stoty/unlabeled_requirement.txt', 'r', encoding='utf-8') as file:
            for line in file.readlines():
                line = line.strip('\n')
                if len(line) == 0:
                    continue
                sentences.append((line, []))
        np.random.shuffle(sentences)
        return sentences

    def build_vocabulary(self):
        """
        构建词典
        :return:
        """
        dict_path = os.path.join('params', '%s_dict.json' %self.training_id)
        if os.path.exists(dict_path):
            with open(dict_path, 'r', encoding='utf-8') as json_file:
                self.vocabulary = json.load(json_file)

        # build from training data
        for (training_input, _) in self.training_data:
            for char_sentence in training_input:
                if char_sentence not in self.vocabulary.keys():
                    self.vocabulary[char_sentence] = max(self.vocabulary.values()) + 1
        # build from pretraining data
        for (sentence, _) in self.pretraining_data:
            for char_sentence in sentence:
                if char_sentence not in self.vocabulary.keys():
                    self.vocabulary[char_sentence] = max(self.vocabulary.values()) + 1

        with open(dict_path, 'w', encoding='utf-8') as json_file:
            json.dump(self.vocabulary, json_file)

    def prepare_batch(self):
        """
        将训练数据划分成batch的形式
        :return:
        """
        index = 0
        while True:
            if index + self.batch_size >= len(self.training_coding.coding_pairs):
                pad_data = self.pad_data(
                    sorted(self.training_coding.coding_pairs[-self.batch_size:],
                           key=(lambda x: len(x[0])), reverse=True)
                )
                self.batch_training_data.append(pad_data)
                break
            else:
                pad_data = self.pad_data(
                    sorted(self.training_coding.coding_pairs[index: index + self.batch_size],
                           key=(lambda x: len(x[0])), reverse=True)
                )
                index += self.batch_size
                self.batch_training_data.append(pad_data)

        if self.experiment_option == 'evaluation':
            index = 0
            while True:
                if index + self.batch_size >= len(self.testing_coding.coding_pairs):
                    pad_data = self.pad_data(
                        sorted(self.testing_coding.coding_pairs[-self.batch_size:],
                               key=(lambda x: len(x[0])), reverse=True), padding_option='global'
                    )
                    self.batch_testing_data.append(pad_data)
                    break
                else:
                    pad_data = self.pad_data(
                        sorted(self.testing_coding.coding_pairs[index: index + self.batch_size],
                               key=(lambda x: len(x[0])), reverse=True), padding_option='global'
                    )
                    index += self.batch_size
                    self.batch_testing_data.append(pad_data)

        if self.is_pretraining:
            index = 0
            while True:
                if index + self.pretraining_batch_size >= len(self.pretraining_coding.coding_pairs):
                    pad_data = self.pad_data(
                        sorted(self.pretraining_coding.coding_pairs[-self.pretraining_batch_size:],
                               key=(lambda x: len(x[0])), reverse=True)
                    )
                    self.batch_pretraining_data.append(pad_data)
                    break
                else:
                    pad_data = self.pad_data(
                        sorted(self.pretraining_coding.coding_pairs[index: index + self.pretraining_batch_size],
                               key=(lambda x: len(x[0])), reverse=True)
                    )
                    index += self.pretraining_batch_size
                    self.batch_pretraining_data.append(pad_data)

        if self.is_pretraining:
            index = 0
            while True:
                if index + self.pretraining_batch_size >= len(self.pretraining_coding.coding_pairs):
                    pad_data = self.pad_data(
                        sorted(self.pretraining_coding.coding_pairs[-self.pretraining_batch_size:],
                               key=(lambda x: len(x[0])), reverse=True)
                    )
                    self.batch_tuning_data.append(pad_data)
                    break
                else:
                    pad_data = self.pad_data(
                        sorted(self.pretraining_coding.coding_pairs[index: index + self.pretraining_batch_size],
                               key=(lambda x: len(x[0])), reverse=True)
                    )
                    index += self.pretraining_batch_size
                    self.batch_tuning_data.append(pad_data)

    def pad_data(self, data, padding_option='local'):
        c_data = copy.deepcopy(data)
        if padding_option == 'local':
            max_length = max([len(i[0]) for i in c_data])
        else:
            max_length = self.max_length
        for i in c_data:
            i.append(len(i[0]))
            i[0] = i[0] + (max_length-len(i[0])) * [0]
            i[1] = i[1] + (max_length-len(i[1])) * [0]
            # i[0] = torch.tensor(i[0])
            # i[1] = torch.tensor(i[1])
        return c_data

    def iteration_training(self):
        idx = 0
        while True:
            yield self.batch_training_data[idx]
            idx += 1
            if idx > len(self.batch_training_data)-1:
                idx = 0

    def get_training_batch(self):
        for data in self.batch_training_data:
            yield data

    def iteration_testing(self):
        idx = 0
        while True:
            yield self.batch_testing_data[idx]
            idx += 1
            if idx > len(self.batch_testing_data)-1:
                idx = 0

    def get_testing_batch(self):
        for data in self.batch_testing_data:
            yield data

    def iteration_pretraining(self):
        idx = 0
        while True:
            yield self.batch_pretraining_data[idx]
            idx += 1
            if idx > len(self.batch_pretraining_data)-1:
                idx = 0

    def get_pretraining_batch(self):
        for data in self.batch_pretraining_data:
            yield data


