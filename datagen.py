import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
import config


def parse_row(row, line_count):
    id, title, citeid, raw_title, abstract = row[0], row[1], row[2], row[3], row[4]
    string_values = config.tags_lines[line_count - 1]
    values_count = int(string_values.split()[0])
    values = string_values.split()[1:]

    return id, title, citeid, raw_title, abstract, values, values_count


### VARIABILITY POINT: how to adjust the score?? how many points scale
def get_adjusted_score(score):
    adj_score = 0
    # low similarity
    if score < 0.3333:
        adj_score = 3

    # medium similarity
    if 0.3333 <= score < 0.6666:
        adj_score = 2

    # high similarity
    if 0.6666 <= score <= 1:
        adj_score = 1
    # print(adj_score)
    return adj_score


def get_score(common_tags, tag_count1, tag_count2):
    try:
        score = common_tags / min(tag_count1, tag_count2)
    except ZeroDivisionError:
        score = 0.0

    adjusted_score = get_adjusted_score(score)

    return adjusted_score


### VARIABILITY POINT: common tags or citacion network??
def tag_based_datagen():
    with open(config.whole_dataset, mode='w') as dataset_file:
        dataset_writer = csv.writer(dataset_file, delimiter=',', quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)

        dataset_writer.writerow(config.header)

    with open(os.path.join(config.data_dir, config.input_raw_data), encoding='ISO-8859-1') as csv_file1:
        csv_reader1 = csv.reader(csv_file1, delimiter=',')
        line_count1 = 0
        for row1 in csv_reader1:
            if line_count1 == 0:
                line_count1 += 1
            else:
                id1, title1, citeid1, raw_title1, abstract1, tags1, tag_count1 = parse_row(row1, line_count1)
                line_count1 += 1

                with open(os.path.join(config.data_dir, config.input_raw_data), encoding='ISO-8859-1') as csv_file2:
                    csv_reader2 = csv.reader(csv_file2, delimiter=',')
                    line_count2 = 0
                    for row2 in csv_reader2:

                        if line_count2 == 0:
                            line_count2 += 1
                        else:
                            id2, title2, citeid2, raw_title2, abstract2, tags2, tag_count2 = parse_row(row2,
                                                                                                       line_count2)

                            # max_len = 512
                            if len(abstract1.split()) + len(raw_title1.split()) + len(abstract2.split()) + len(
                                    raw_title2.split()) > 260:
                                continue

                            common_tags = len(list(set(tags1).intersection(set(tags2))))

                            score = get_score(common_tags, tag_count1, tag_count2)

                            with open(config.whole_dataset, mode='a') as dataset_file:
                                dataset_writer = csv.writer(dataset_file, delimiter=',', quotechar='"',
                                                            quoting=csv.QUOTE_MINIMAL)

                                dataset_writer.writerow(
                                    [id1, id2, title1 + '. ' + abstract1.lower(), title2 + '. ' + abstract2.lower(),
                                     score])

                            line_count2 += 1


def citation_based_datagen():
    with open(config.whole_dataset_cit, mode='w') as dataset_file:
        dataset_writer = csv.writer(dataset_file, delimiter=',', quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)

        dataset_writer.writerow(config.header)

    with open(os.path.join(config.data_dir, config.input_raw_data), encoding='ISO-8859-1') as csv_file1:
        csv_reader1 = csv.reader(csv_file1, delimiter=',')
        line_count1 = 0
        for row1 in csv_reader1:
            if line_count1 == 0:
                line_count1 += 1
            else:
                id1, title1, citeid1, raw_title1, abstract1, cits1, cits_count1 = parse_row(row1, line_count1)
                line_count1 += 1

                with open(os.path.join(config.data_dir, config.input_raw_data), encoding='ISO-8859-1') as csv_file2:
                    csv_reader2 = csv.reader(csv_file2, delimiter=',')
                    line_count2 = 0
                    for row2 in csv_reader2:

                        if line_count2 == 0:
                            line_count2 += 1
                        else:
                            id2, title2, citeid2, raw_title2, abstract2, cits2, cits_count2 = parse_row(row2,
                                                                                                        line_count2)

                            if len(abstract1.split()) > 512 or len(abstract2.split()) > 512:
                                continue

                            common_tags = len(list(set(cits1).intersection(set(cits2))))

                            score = get_score(common_tags, cits_count1, cits_count2)

                            with open(config.whole_dataset_cit, mode='a') as dataset_file:
                                dataset_writer = csv.writer(dataset_file, delimiter=',', quotechar='"',
                                                            quoting=csv.QUOTE_MINIMAL)

                                dataset_writer.writerow(
                                    [id1, id2, title1 + '. ' + abstract1.lower(), title2 + '. ' + abstract2.lower(),
                                     score])

                            line_count2 += 1


def split_dataset():
    dataframe = pd.read_csv('shuf_dataset.csv', sep=',', names=config.header)
    dataframe['split'] = np.random.randn(dataframe.shape[0], 1)

    print(dataframe.head(10))

    msk = np.random.rand(len(dataframe)) <= 0.7

    another = dataframe[msk]
    test = dataframe[~msk]

    another['split_1'] = np.random.randn(another.shape[0], 1)
    take = np.random.rand(len(another)) <= 0.8

    train = another[take]
    dev = another[~take]

    train.to_csv(config.train_path, sep=',', columns=config.header)
    test.to_csv(config.test_path, sep=',', columns=config.header)
    dev.to_csv(config.dev_path, sep=',', columns=config.header)

    # print(train.head(10))


def encode_sentence(tokenizer, s):
    tokens = list(tokenizer.tokenize(s))
    tokens.append('[SEP]')

    return tokenizer.convert_tokens_to_ids(tokens)


def bert_encode(data, tokenizer):
    sentence1 = tf.ragged.constant([
        encode_sentence(tokenizer, s) for s in data["sentence1"]])

    sentence2 = tf.ragged.constant([
        encode_sentence(tokenizer, s) for s in data["sentence2"]])

    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])] * sentence1.shape[0]
    input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)

    input_mask = tf.ones_like(input_word_ids).to_tensor()

    type_cls = tf.zeros_like(cls)
    type_s1 = tf.zeros_like(sentence1)
    type_s2 = tf.ones_like(sentence2)
    input_type_ids = tf.concat(
        [type_cls, type_s1, type_s2], axis=-1).to_tensor()

    print(type_cls.shape, type_s1.shape, type_s2.shape, input_type_ids.shape)

    inputs = {
        'input_word_ids': input_word_ids.to_tensor(),
        'input_mask': input_mask,
        'input_type_ids': input_type_ids}

    return inputs


def prepare_data():
    train_df = pd.read_csv(config.train_path)
    dev_df = pd.read_csv(config.dev_path)
    test_df = pd.read_csv(config.test_path)

    print(train_df.head(1))
    # Set up tokenizer to generate Tensorflow dataset
    tokenizer = config.tokenizer

    tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])

    train = bert_encode(train_df, tokenizer)
    train_labels_df = train_df['score'].map(lambda x: x[0].isdigit())
    train_labels = train_labels_df.astype('int64')

    validation = bert_encode(dev_df, tokenizer)
    validation_labels = dev_df['score'].astype('int64')

    test = bert_encode(test_df, tokenizer)
    test_labels = test_df['score'].astype('int64')

    for key, value in test.items():
        print(f'{key:15s} shape: {value.shape}')

    print(f'glue_train_labels shape: {test_labels.shape}')

    return test, test_labels, train, train_labels, validation, validation_labels
