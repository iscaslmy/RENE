import torch
import torch.optim as optim
import torch.nn as nn
import csv
import sys
from torch.nn.utils import clip_grad_norm_
import numpy as np
import os
import yaml

from LSTM_CRF import LSTM_CRF
from LSTM_LM import LSTM_LM
from data_manager import DataManager
from utils import f1_score
from utils import decode_one_hot_sequence
from utils import dump_keyword_oriented_corpus
from logger import Logger


config_file = open('config.yml', 'r', encoding='utf-8')
params_loader = yaml.load(config_file, Loader=yaml.FullLoader)
detail_tag = params_loader.get('training_id')

sys.stdout = Logger('logs/' + detail_tag, sys.stdout)
sys.stderr = Logger('logs/' + detail_tag, sys.stderr)		# redirect std err, if necessary

torch.manual_seed(2)

START_TAG = "<START>"
STOP_TAG = "<STOP>"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Knowledge_Extract(object):
    def __init__(self,
                 training_id,
                 actions,
                 key_glossary_params,
                 key_pretraining_params,
                 embedding_dim=100,
                 hidden_dim=128,
                 coding_level='char',
                 experiment_option='evaluation',
                 model_saved_path='params/'):
        # parameters dict
        self.key_glossary_params = key_glossary_params
        self.key_pretraining_params = key_pretraining_params
        self.actions = actions

        # constant common parameters
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.coding_level = coding_level
        self.experiment_option = experiment_option
        self.training_id = training_id

        print('DATA LOADING')
        # experimental data
        self.data_manager = DataManager(training_id=self.training_id,
                                        experiment_option=self.experiment_option,
                                        pretraining_batch_size=self.key_pretraining_params.get('batch_size', 20),
                                        coding_level=self.coding_level)
        self.training_batches = self.data_manager.iteration_training()
        self.testing_batches = self.data_manager.iteration_testing()
        self.pretraining_batches = self.data_manager.iteration_pretraining()
        self.total_size = len(self.data_manager.batch_training_data)
        print('DATA LOADING FINISHED!!!')

        # pretraining model
        self.lm_model = LSTM_LM(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            vocab_size=len(self.data_manager.vocabulary),
            batch_size=self.key_pretraining_params.get('batch_size', 20)
        ).to(device)

        # model
        self.model = LSTM_CRF(
            vocab_size=len(self.data_manager.vocabulary),
            tag_to_ix={"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4},
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            batch_size=self.key_glossary_params.get('batch_size', 20),
            dropout=self.key_glossary_params.get('dropout', 0.5)
        ).to(device)
        self.model_saved_path = model_saved_path

    def train(self):
        """
        模型训练
        :return:
        """
        """"
        2.1 PRE_TRAINING LANGUAGE MODEL
        """
        print('PRE_TRAINING MODEL')
        # 损失构建与优化
        criterion = nn.CrossEntropyLoss()  # 交叉熵损失
        # 使用Adam优化方法 最小化损失 优化更新模型参数
        optimizer = torch.optim.Adam(self.lm_model.parameters(),
                                     lr=self.key_pretraining_params.get('learning_rate', 0.002))
        pretraining_epoch_num = self.key_pretraining_params.get('epoch_num', 5)\
            if self.key_pretraining_params.get('is_pretraining', False) and 'pretraining'in self.actions else 0

        for epoch in range(pretraining_epoch_num):
            if os.path.exists('model/lm_model' + detail_tag + '.plk'):
                pretrained_dict = torch.load('model/lm_model' + detail_tag + '.plk')
                self.lm_model.load_state_dict(pretrained_dict)
            # 训练模型
            index = 0

            for batch in self.data_manager.batch_pretraining_data:
                index += 1
                sentences, tags, length = zip(*batch)
                sentences_tensor = torch.tensor(sentences, dtype=torch.long).to(device)
                tags_tensor = torch.tensor(tags, dtype=torch.long).to(device)
                # 前向运算
                # states = detach(states)
                outputs = self.lm_model(sentences_tensor)
                loss = criterion(outputs, tags_tensor.reshape(-1))

                if index % 200 == 1:
                    print('全量数据迭代轮次 [{}/{}], Step数[{}/{}], 损失Loss: {:.4f}, 困惑度/Perplexity: {:5.2f}'\
                                .format(epoch + 1, pretraining_epoch_num, index,
                                len(self.data_manager.batch_pretraining_data),
                                loss.item(), np.exp(loss.item())))

                # 反向传播与优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                clip_grad_norm_(self.lm_model.parameters(), 0.5)

                del loss
                del outputs
                del sentences_tensor
                del tags_tensor

            torch.save(self.lm_model.state_dict(), 'model/lm_model' + detail_tag + '.plk')
            torch.cuda.empty_cache()
        # self.lm_model.load_state_dict(torch.load('model/lm_model.pth'))
        print('PRE_TRAIN LANGUAGE MODEL FINISHED!!!')

        """"
        2.2 FINE_TUNING LANGUAGE MODEL
        """
        print('PRE_TRAINING MODEL')
        # 损失构建与优化
        criterion = nn.CrossEntropyLoss()  # 交叉熵损失
        # 使用Adam优化方法 最小化损失 优化更新模型参数
        optimizer = torch.optim.Adam(self.lm_model.parameters(),
                                     lr=self.key_pretraining_params.get('learning_rate', 0.001))
        pretraining_epoch_num = self.key_pretraining_params.get('epoch_num', 5) \
            if self.key_pretraining_params.get('is_pretraining', False) and 'pretraining' in self.actions else 0

        for epoch in range(pretraining_epoch_num):
            if os.path.exists('model/lm_model' + detail_tag + '.plk'):
                pretrained_dict = torch.load('model/lm_model' + detail_tag + '.plk')
                self.lm_model.load_state_dict(pretrained_dict)
            # 训练模型
            index = 0

            for batch in self.data_manager.batch_tuning_data:
                index += 1
                sentences, tags, length = zip(*batch)
                sentences_tensor = torch.tensor(sentences, dtype=torch.long).to(device)
                tags_tensor = torch.tensor(tags, dtype=torch.long).to(device)
                # 前向运算
                # states = detach(states)
                outputs = self.lm_model(sentences_tensor)
                loss = criterion(outputs, tags_tensor.reshape(-1))

                if index % 200 == 1:
                    print('全量数据迭代轮次 [{}/{}], Step数[{}/{}], 损失Loss: {:.4f}, 困惑度/Perplexity: {:5.2f}' \
                          .format(epoch + 1, pretraining_epoch_num, index,
                                  len(self.data_manager.batch_pretraining_data),
                                  loss.item(), np.exp(loss.item())))

                # 反向传播与优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                clip_grad_norm_(self.lm_model.parameters(), 0.5)

                del loss
                del outputs
                del sentences_tensor
                del tags_tensor

            torch.save(self.lm_model.state_dict(), 'model/lm_model' + detail_tag + '.plk')
            torch.cuda.empty_cache()
        # self.lm_model.load_state_dict(torch.load('model/lm_model.pth'))
        print('FINE_TUNING LANGUAGE MODEL FINISHED!!!')
        """
        LOAD THE PARAMETERS INTO TARGET MODEL
        """
        # load pretrained params
        if os.path.exists('model/lm_model' + detail_tag + '.plk'):
            glossary_dict = self.model.state_dict()
            pretrained_dict = torch.load('model/lm_model' + detail_tag + '.plk')
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in glossary_dict}
            glossary_dict.update(pretrained_dict)
            self.model.load_state_dict(glossary_dict)
            # frozen the parameters
            frozen_list = [x for x in list(glossary_dict.keys() & pretrained_dict.keys()) if 'word_embeds' in x]
            for k, v in self.model.named_parameters():
                if k in frozen_list:
                    v.requires_grad = False

        """
        FINE_TURING THE MODEL
        """
        print('FINE_TURNING MODEL')
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                              lr=self.key_glossary_params.get('learning_rate', 0.001),
                              weight_decay=float(self.key_glossary_params.get('weight_decay', 1e-4)))

        num_epoch = self.key_glossary_params.get('epoch_num', 30) if 'glossary' in self.actions else 0
        for epoch in range(num_epoch):
            if (epoch+1) % 50 == 0:
                for p in optimizer.param_groups:
                    p['lr'] *= 0.1
            # lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
            index = 0
            for batch in self.data_manager.batch_training_data:
                index += 1
                optimizer.zero_grad()

                sentences, tags, length = zip(*batch)
                sentences_tensor = torch.tensor(sentences, dtype=torch.long).to(device)
                tags_tensor = torch.tensor(tags, dtype=torch.long).to(device)
                length_tensor = torch.tensor(length, dtype=torch.long).to(device)

                loss = self.model.neg_log_likelihood(sentences_tensor, tags_tensor, length_tensor)
                progress = ("█" * int(index * 25 / self.total_size)).ljust(25)
                print("""epoch [{}] |{}| {}/{}\n\tloss {:.2f}""".format(
                    epoch, progress, index, self.total_size, loss.cpu().tolist()[0]
                )
                )
                self.evaluate()
                print("-" * 50)
                loss.backward()
                optimizer.step()
                torch.save(self.model.state_dict(), self.model_saved_path + 'params.pkl')

            print("%" * 80)
            self.evaluate_all()
            print("%" * 80)
            csv_rows = []
            all_test_samples = []
            for batch in self.data_manager.batch_testing_data:
                all_test_samples.extend(batch)
            sentences, tar_paths, length = zip(*all_test_samples)
            _, pre_paths = self.model(sentences)
            reversed_vocabulary = {v: k for k, v in self.data_manager.training_coding.vocabulary.items()}
            for fetch in zip(tar_paths, pre_paths, sentences):
                tar, pre, sentence = fetch
                sentence_list = [reversed_vocabulary[x] for x in sentence]
                sentence_list = [x for x in sentence_list if x != 'unk']
                sentence_contents = ''.join(sentence_list)
                tar_contents = decode_one_hot_sequence(sentence, tar, self.data_manager.training_coding.vocabulary)
                pre_contents = [x for x in decode_one_hot_sequence(sentence, pre, self.data_manager.training_coding.vocabulary) if len(x) > 0]
                csv_rows.append([sentence_contents, tar_contents, pre_contents])

            with open('details/' + detail_tag + '_details.csv', 'a', newline='', encoding='utf-8') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(['******************************EPOCH %s******************************' % str(epoch)])
                for data in csv_rows:
                    csv_writer.writerow(data)

    def evaluate(self):
        """
        模型评估函数
        :return:
        """
        sentences, tar_paths, length = zip(*self.testing_batches.__next__())
        _, pre_paths = self.model(sentences)
        print("\teval")
        f1_score(tar_paths, pre_paths, sentences, self.data_manager.training_coding.vocabulary)

    def evaluate_all(self):
        """
        模型评估函数
        :return:
        """
        all_test_samples = []
        for batch in self.data_manager.batch_testing_data:
            all_test_samples.extend(batch)
        sentences, tar_paths, length = zip(*all_test_samples)
        _, pre_paths = self.model(sentences)
        print("\teval")
        f1_score(tar_paths, pre_paths, sentences, self.data_manager.training_coding.vocabulary)


def run():
    global params_loader
    training_id = params_loader.get('training_id')
    glossary_params = params_loader.get('glossary_extracting', dict())
    pretraining_params = params_loader.get('pretraining', dict())
    embedding_dim = params_loader.get('embedding_dim', 128)
    hidden_dim = params_loader.get('hidden_dim', 1024)
    print(pretraining_params)
    print(glossary_params)
    print(embedding_dim, hidden_dim)

    actions = params_loader.get('actions', ['pretraining, glossary'])

    pretraining_corpus_path = pretraining_params['corpus_path']
    pretraining_keywords = pretraining_params['keywords']
    if not os.path.exists('wiki_data/pretraining_corpus.txt'):
        print('DUMPING WIKI CONCEPTS')
        dump_keyword_oriented_corpus(target=pretraining_corpus_path, wiki_keywords=pretraining_keywords)
        print('DUMPING WIKI CONCEPTS FINISHED!!!')

    ie = Knowledge_Extract(
        training_id=training_id,
        embedding_dim=embedding_dim, hidden_dim=hidden_dim,
        key_glossary_params=glossary_params, key_pretraining_params=pretraining_params,
        actions=actions
    )
    ie.train()


if __name__ == "__main__":
    run()


