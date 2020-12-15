import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
import config


def parse_row(row, line_count, other):
    id, title, citeid, raw_title, abstract = row[0], row[1], row[2], row[3], row[4]
    string_values = other[line_count - 1]
    values_count = int(string_values.split()[0])
    values = string_values.split()[1:]

    return id, title, citeid, raw_title, abstract, values, values_count


### VARIABILITY POINT: how to adjust the score?? how many points scale
# HOW MUCH THIS CHOICE INFLUENCES THE PERFORMANCE??
# 3 values score creates a majority of 3-labeled examples, seems that the model overfits and tends to label as 3 all the validation examples
# seems like a binary score is much better
def get_adjusted_score(score):
    adj_score = 0
    # low similarity
    # if score < 0.3333:
    #    adj_score = 0

    # medium similarity
    # if 0.3333 <= score < 0.6666:
    #    adj_score = 1

    # high similarity
    # if 0.6666 <= score <= 1:
    #    adj_score = 2
    # print(adj_score)
    if score >= 0.5000:
        adj_score = 1

    return adj_score


def get_score(common_tags, tag_count1, tag_count2):
    try:
        score = common_tags / min(tag_count1, tag_count2)
    except ZeroDivisionError:
        score = 0.0

    adjusted_score = get_adjusted_score(score)

    return adjusted_score


### VARIABILITY POINT: common tags or citacion network??
def do_datagen(input_file, output_file, other):
    with open(os.path.join(config.data_dir, input_file), encoding='ISO-8859-1') as csv_file1:
        csv_reader1 = csv.reader(csv_file1, delimiter=',')
        line_count1 = 0
        for row1 in csv_reader1:
            if line_count1 == 0:
                line_count1 += 1
            else:
                id1, title1, citeid1, raw_title1, abstract1, tags1, tag_count1 = parse_row(row1, line_count1, other)
                line_count1 += 1

                with open(os.path.join(config.data_dir, input_file), encoding='ISO-8859-1') as csv_file2:
                    csv_reader2 = csv.reader(csv_file2, delimiter=',')
                    line_count2 = 0
                    for row2 in csv_reader2:

                        if line_count2 == 0:
                            line_count2 += 1
                        else:
                            id2, title2, citeid2, raw_title2, abstract2, tags2, tag_count2 = parse_row(row2,
                                                                                                       line_count2, other)

                            common_tags = len(list(set(tags1).intersection(set(tags2))))

                            score = get_score(common_tags, tag_count1, tag_count2)

                            with open(output_file, mode='a') as dataset_file:
                                dataset_writer = csv.writer(dataset_file, delimiter=',', quotechar='"',
                                                            quoting=csv.QUOTE_MINIMAL)

                                dataset_writer.writerow(
                                    [id1, id2, title1 + '. ' + abstract1.lower(), title2 + '. ' + abstract2.lower(),
                                     score])

                            line_count2 += 1


def split_dataset(input_file, generation_type):
    dataframe = pd.read_csv(input_file, sep=',', names=config.header)
    dataframe['split'] = np.random.randn(dataframe.shape[0], 1)

    print(dataframe.head(10))

    msk = np.random.rand(len(dataframe)) <= 0.7

    another = dataframe[msk]
    test = dataframe[~msk]

    another['split_1'] = np.random.randn(another.shape[0], 1)
    take = np.random.rand(len(another)) <= 0.8

    train = another[take]
    dev = another[~take]

    if generation_type == 'TAG':
        train.to_csv(config.train_path, sep=',')
        test.to_csv(config.test_path, sep=',')
        dev.to_csv(config.dev_path, sep=',')

    else:
        train.to_csv(config.train_path_cit, sep=',')
        test.to_csv(config.test_path_cit, sep=',')
        dev.to_csv(config.dev_path_cit, sep=',')


def encode_sentence(tokenizer, s):
    toks = list(tokenizer.tokenize(s))
    tokens = toks

    # truncate long sentences to a max of 254 --> 254*2 + 3(2 sep and cls) = 511 sequence length
    if len(toks) > 254:
        tokens = toks[:254]

    tokens.append('[SEP]')

    return tokenizer.convert_tokens_to_ids(tokens)


def bert_encode(data, tokenizer):
    sentence1 = tf.ragged.constant([
        encode_sentence(tokenizer, s) for s in data["sentence1"]])

    sentence2 = tf.ragged.constant([
        encode_sentence(tokenizer, s) for s in data["sentence2"]])

    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])] * sentence1.shape[0]
    input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)

    # WHY ??
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


def prepare_data(keyword):
    if keyword == 'TAG':
        train_df = pd.read_csv(config.train_path, sep=',', names=config.header_split, skiprows=1)
        dev_df = pd.read_csv(config.dev_path, sep=',', names=config.header_split, skiprows=1)
        test_df = pd.read_csv(config.test_path, sep=',', names=config.header_split, skiprows=1)

    else:
        train_df = pd.read_csv(config.train_path_cit, sep=',', names=config.header_split, skiprows=1)
        dev_df = pd.read_csv(config.dev_path_cit, sep=',', names=config.header_split, skiprows=1)
        test_df = pd.read_csv(config.test_path_cit, sep=',', names=config.header_split, skiprows=1)

    # Set up tokenizer to generate Tensorflow dataset
    tokenizer = config.tokenizer

    train = bert_encode(train_df, tokenizer)
    # train_labels_df = train_df['score'].map(lambda x: x != 'score')
    train_labels = train_df['score'].astype('int64')

    validation = bert_encode(dev_df, tokenizer)
    validation_labels = dev_df['score'].astype('int64')

    test = bert_encode(test_df, tokenizer)
    test_labels = test_df['score'].astype('int64')

    for key, value in train.items():
        print(f'{key:15s} shape: {value.shape}')

    print(f'train_labels shape: {train_labels.shape}')

    return test, test_labels, train, train_labels, validation, validation_labels


"""
def shuffle_and_write(command, shuffled):
    os.system(command)
    os.system('cat shuffled/app.csv >> ' + shuffled)
    os.system(config.command_remove_app)
"""
