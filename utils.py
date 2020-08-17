import distance
from opencc import OpenCC
import os
import json
import re
import csv


# 反向传播过程“截断”(不复制gradient)
def detach(states):
    return [state.detach() for state in states]


def f1_score(tar_paths, pre_paths, sentences, vocab):
    """
    计算指标值
    :param tar_paths:
    :param pre_paths:
    :param sentences:
    :param vocab:
    :return:
    """
    origin = 0.
    found = 0.
    right = 0.
    for fetch in zip(tar_paths, pre_paths, sentences):
        tar, pre, sentence = fetch
        tar_contents = decode_one_hot_sequence(sentence, tar, vocab)
        pre_contents = [x for x in decode_one_hot_sequence(sentence, pre, vocab) if len(x) > 0]
        # print(tar_contents, pre_contents)

        # str = ''
        # for pre_tag in pre_tags:
        #     for index in pre_tag:
        #         for content, voca_index in vocab.items():
        #             if voca_index == index:
        #                 str += content
        #                 break
        # print(str)

        origin += len(tar_contents)
        found += len(pre_contents)

        for p_tag in pre_contents:
            for t_tag in tar_contents:
                if 1 - distance.nlevenshtein(p_tag, t_tag, method=1) >= 0.6:
                    right += 1
                    break

    recall = 0. if origin == 0 else (right / origin)
    precision = 0. if found == 0 else (right / found)
    f1 = 0. if recall+precision == 0 else (2*precision*recall)/(precision + recall)
    print("\trecall {:.2f}\tprecision {:.2f}\tf1 {:.2f}".format(recall, precision, f1))
    return recall, precision, f1


def decode_one_hot_sequence(word_sequence, label_sequence, vocabulary):
    """
    给出编码，根据字典解码为字符串
    :param word_sequence: 特征编码
    :param label_sequence: 标签编码
    :param vocabulary: 词典
    :return:  []
    """
    index_intervals = []
    start_index = -1
    for i, label in enumerate(label_sequence):
        if label == 0:
            start_index = i
        if label == 1:
            if i == len(label_sequence)-1 or label_sequence[i+1] == 2:
                end_index = i
                if start_index != -1:
                    index_intervals.append((start_index, end_index+1))
                start_index = -1

    results = []
    reversed_vocabulary = {v: k for k, v in vocabulary.items()}
    for start_index, end_index in index_intervals:
        word_index_sequence = word_sequence[start_index: end_index]
        result = [reversed_vocabulary.get(word_index, 0) for word_index in word_index_sequence]
        result = [x for x in result if x != 'unk']
        results.append(''.join(result))

    # str = [reversed_vocabulary.get(word_index, 0) for word_index in word_sequence]
    # print(''.join(str))

    return list(set(results))


def dump_wiki_dict(file_path, dict_name='concepts_dict', rebuild=False):
    """
    按行读入txt文件文本内容 并转化为中文
    :param file_path:  str 文本路径
    :param dict_name: 保存的字典名称
    :param rebuild: 是否重新构建词典
    :return: [str, str] 文本
    """
    if rebuild or not os.path.exists('wiki_data/' + dict_name):
        opencc = OpenCC('t2s')
        wiki_dict = dict()
        with open(file_path, encoding='utf-8') as file_reader:
            lines = file_reader.readlines()
            for line in lines:
                line = opencc.convert(line)
                splitted_line = line.split(':')
                concept = splitted_line[-1]
                index = splitted_line[: -1]
                wiki_dict[concept] = ':'.join(index)
            with open('wiki_data/' + dict_name, 'w', encoding='utf-8') as f:
                json.dump(wiki_dict, f)
            return wiki_dict
    else:
        with open('wiki_data/' + dict_name, 'r', encoding='utf-8') as f:
            return json.load(f)


def text_preprocessing(text):
    """
    预处理文本，
    :param text:
    :return:
    """
    p1 = re.compile('（）')
    p2 = re.compile('《》')
    p3 = re.compile('「')
    p4 = re.compile('」')
    # p5 = re.compile(r'[')
    # p6 = re.compile(']')
    p7 = re.compile('【')
    p8 = re.compile('】')
    # p5 = re.compile('<doc (.*)>')
    # p6 = re.compile('</doc>')
    text = p1.sub('', text)
    text = p2.sub('', text)
    text = p3.sub('', text)
    text = p4.sub('', text)
    # text = p5.sub('', text)
    # text = p6.sub('', text)
    text = p7.sub('', text)
    text = p8.sub('', text)
    # text = p5.sub('', text)
    # text = p6.sub('', text)
    return text


def dump_wiki_concepts(file_parent_path='wiki_data/articles'):
    """
    解析文件内容，获取计算机相关的词典
    :param file_parent_path:
    :return:
    """
    article_elements = dict()

    opencc = OpenCC('t2s')
    start_pattern = re.compile('<doc (.*)>')
    end_pattern = re.compile('</doc>')
    for file_name in os.listdir(file_parent_path):
        file_path = os.path.join(file_parent_path, file_name)
        print(file_path)

        id = ''
        title = ''
        contents = []
        with open(file_path, 'r', encoding='utf-8') as wiki_file:
            for line in wiki_file.readlines():
                if len(line) == 0:
                    continue
                if re.match(start_pattern, line):
                    id_pattern = re.compile('id=\"\d+\"')
                    title_pattern = re.compile('title=\"(.*)\"')
                    id = id_pattern.findall(line)[0][4: -1]
                    title = title_pattern.findall(line)[0]
                elif re.match(end_pattern, line):
                    article_elements[id] = (title, ''.join(contents))
                    id = ''
                    title = ''
                    contents = []
                else:
                    contents.append(opencc.convert(text_preprocessing(line)))

        with open('wiki_data/dumped_wiki.json', 'w', encoding='utf-8') as f:
            json.dump(article_elements, f)


def dump_keyword_oriented_corpus(source='wiki_data/dumped_wiki.json',
                             target='wiki_data/pretraining_corpus.txt',
                             wiki_keywords=['软件', '数据库', '计算机', '计算机网络',
                                            '金融', '经济', '银行', '证券', '保险', '基金', '投资']):
    """
    dump相关的文本
    :param source:
    :param target:
    :param wiki_keywords:
    :return:
    """
    opencc = OpenCC('t2s')

    if not os.path.exists(source):
        dump_wiki_concepts(source)

    with open(source, 'r', encoding='utf-8', ) as f:
        article_elements = json.load(f)

    candidates = []
    for id, (title, text) in article_elements.items():
        if any(category in text for category in wiki_keywords):
            candidates.append([id, opencc.convert(title)])

    with open('wiki_data/keyword_oriented_index.csv', 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        for data in candidates:
            csv_writer.writerow(data)

    txt_lines = []
    corpus = dict()
    ids = [element[0] for element in candidates]
    for id, (title, text) in article_elements.items():
        if id not in ids:
            continue
        corpus[id] = (title, text)
        sentences = text_preprocessing(text).split('。')
        for sentence in sentences:
            if len(sentence.strip()) == 0:
                continue
            txt_lines.append(sentence.strip())

    # with open(target, 'w', encoding='utf-8') as json_file:
    #     json.dump(corpus, json_file)
    with open(target, 'w', encoding='utf-8') as corpus_txt:
        for line in txt_lines:
            corpus_txt.write(line + '\n')


if __name__ == "__main__":
    dump_keyword_oriented_corpus()


