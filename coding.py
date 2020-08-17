"""
数据编码、解码
"""
import jieba.posseg as psg

from requirement_reader import read_all_labeled_stories


START_TAG = "<START>"
STOP_TAG = "<STOP>"


class Coding:
    def __init__(self,
                 vocabulary,
                 coding_level='char',
                 original_pairs=[],
                 max_input_length=100,
                 is_pretraining=False):
        self.coding_level = coding_level
        self.is_pretraining = is_pretraining

        self.max_input_length = max_input_length

        self.vocabulary = vocabulary
        self.tag_map = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

        self.original_pairs = original_pairs
        self.coding_pairs = []

        # approaches invoked
        self.tag_labeled_stories()

    def tag_labeled_stories(self):
        """
        给所有的标注故事打标签
        :return:
        """
        if self.coding_level == 'char':
            self.__encode_training_at_char_level()
        else:
            self.__encode_training_at_word_level()

    def __encode_training_at_char_level(self):
        """
        字符级编码
        :return:
        """
        max_length = -1
        for (sentence, ground_truths) in self.original_pairs:
            if len(sentence) > max_length:
                max_length = len(sentence)
            sentence_encoding = []
            tags = []
            ground_truths.sort(key=lambda i: len(i), reverse=True)
            index_intervals = []
            for ground_truth in ground_truths:
                for index_sentence, char_sentence in enumerate(sentence):
                    window_str = sentence[index_sentence: index_sentence+len(ground_truth)]
                    if window_str == ground_truth:
                        index_intervals.append((index_sentence, index_sentence+len(ground_truth)))

            # for sentences
            for char_sentence in sentence:
                # if char_sentence not in self.vocabulary.keys():
                #     self.vocabulary[char_sentence] = max(self.vocabulary.values()) + 1
                sentence_encoding.append(self.vocabulary.get(char_sentence, 0))
            # if self.is_pretraining:
            #     sentence_encoding.append(-1)

            # for tags
            if self.is_pretraining:
                tags = sentence_encoding[1:]
            else:
                for _ in range(len(sentence)):
                    tags.append('O')
                for index_interval in index_intervals:
                    tags[index_interval[0]] = 'B'
                    for i in range(index_interval[0]+1, index_interval[1]):
                        tags[i] = 'I'
                tags = [self.tag_map.get(x, 0) for x in tags]

            self.coding_pairs.append([sentence_encoding, tags])

        self.max_input_length = max_length

    def __encode_training_at_word_level(self):
        """
        单词级编码
        :return:
        """
        max_length = -1
        for (sentence, ground_truths) in self.original_pairs:
            sentence_encoding = []
            tags = []

            index_intervals = []
            cutted_sentences = [x.word for x in psg.cut(sentence)]
            if len(cutted_sentences) > max_length:
                max_length = len(cutted_sentences)
            ground_truths.sort(key=lambda i:len(i),reverse=True)
            for ground_truth in ground_truths:
                cutted_groundtruths = [x.word for x in psg.cut(ground_truth)]
                for index_sentence, word_sentence in enumerate(cutted_sentences):
                    window_str = ''.join(cutted_sentences[index_sentence: index_sentence + len(cutted_groundtruths)])
                    if window_str == ground_truth:
                        index_intervals.append((index_sentence, index_sentence + len(cutted_groundtruths)))

            # for sentences
            for word_sentence in cutted_sentences:
                # if word_sentence not in self.vocabulary.keys():
                #     self.vocabulary[word_sentence] = max(self.vocabulary.values()) + 1
                sentence_encoding.append(self.vocabulary.get(word_sentence, 0))

            # for tags
            for _ in range(len(cutted_sentences)):
                tags.append('O')
            for index_interval in index_intervals:
                tags[index_interval[0]] = 'B'
                for i in range(index_interval[0]+1, index_interval[1]):
                    tags[i] = 'I'
            tags = [self.tag_map.get(x, 0) for x in tags]

            self.coding_pairs.append([sentence_encoding, tags])

        self.max_input_length = max_length

    def decode_one_hot_sequence(self, word_sequence, label_sequence):
        """
        给出编码，根据字典解码为字符串
        :param word_sequence: 特征编码
        :param label_sequence: 标签编码
        :return:  []
        """
        assert len(self.vocabulary) > 1

        index_intervals = []
        start_index = -1
        # end_index = -1
        for i, label in enumerate(label_sequence):
            if label == 'B':
                start_index = i
            if label == 'I':
                if i == len(label_sequence)-1 or label_sequence[i+1] == 'O':
                    end_index = i
                    if start_index != -1:
                        index_intervals.append((start_index, end_index+1))
                    start_index = -1
                    # end_index = -1

        results = []
        reversed_vocabulary = {v: k for k, v in self.vocabulary.items()}
        for start_index, end_index in index_intervals:
            word_index_sequence = word_sequence[start_index: end_index]
            result = [reversed_vocabulary[word_index] for word_index in word_index_sequence]
            results.append(''.join(result))

        return results

    def encode_into_one_hot(self, word_sequence):
        """
        讲特征序列进行one-hot编码
        :param word_sequence:
        :return: one-hot编码序列[]
        """
        assert len(self.vocabulary) > 1

        feature_one_hot = [self.vocabulary[word] for word in word_sequence]
        return feature_one_hot

