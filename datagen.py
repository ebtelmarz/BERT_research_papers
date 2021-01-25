import os
import re
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

    return id, title.strip(), citeid, raw_title.strip(), abstract.strip(), values, values_count


### VARIABILITY POINT: how to adjust the score?? how many points scale
# HOW MUCH THIS CHOICE INFLUENCES THE PERFORMANCE??
# 3 values score creates a majority of 3-labeled examples, seems that the model overfits and tends to label as 3 all the validation examples
# seems like a binary score is much better
def get_adjusted_score(score):
    adj_score = 0
    """
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
   
    if 0.0 <= score < 0.2:
        adj_score = 0

    if 0.2 <= score < 0.4:
        adj_score = 1

    if 0.4 <= score < 0.6:
        adj_score = 2

    if 0.6 <= score < 0.8:
        adj_score = 3

    if 0.8 <= score <= 1:
        adj_score = 4
    """
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


### VARIABILITY POINT: common tags or citation network??
def clean_text(text):
    stripped = re.sub('\[[^>]+\]', '', text)

    cleaned_words = [word.replace('{', '') for word in stripped.split()]
    cleaned_words = [word.replace('}', '') for word in cleaned_words]
    # cleaned_words = [word for word in cleaned_words if word.isalnum()]
    cleaned_text = ' '.join(cleaned_words).lower()

    # cleaned_text = stripped.lower()

    return cleaned_text


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

                abs1 = clean_text(abstract1)

                with open(os.path.join(config.data_dir, input_file), encoding='ISO-8859-1') as csv_file2:
                    csv_reader2 = csv.reader(csv_file2, delimiter=',')
                    line_count2 = 0
                    for row2 in csv_reader2:

                        if line_count2 == 0:
                            line_count2 += 1
                        else:
                            id2, title2, citeid2, raw_title2, abstract2, tags2, tag_count2 = parse_row(row2, line_count2, other)

                            common_tags = len(list(set(tags1).intersection(set(tags2))))
                            # common_tags_union = len(list(set(tags1).union(set(tags2))))
                            score = get_score(common_tags, tag_count1, tag_count2)

                            abs2 = clean_text(abstract2)

                            with open(output_file, mode='a') as dataset_file:
                                dataset_writer = csv.writer(dataset_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                dataset_writer.writerow([id1, id2, title1 + '. ' + abs1, title2 + '. ' + abs2, score])

                            line_count2 += 1


def split_dataset(input_file):
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

    train.to_csv(config.train_path, sep=',')
    test.to_csv(config.test_path, sep=',')
    dev.to_csv(config.dev_path, sep=',')


def encode_sentence(tokenizer, s):
    sent1 = str(s[0])
    sent2 = str(s[1])

    tokens1 = list(tokenizer.tokenize(sent1))
    tokens2 = list(tokenizer.tokenize(sent2))

    sentence1_length = len(tokens1)
    sentence2_length = len(tokens2)

    if sentence1_length + sentence2_length + 3 > 512:
        if sentence1_length > 254:
            if sentence2_length > 254:
                tokens1 = tokens1[:253]
                tokens2 = tokens2[:253]
            else:
                difference = 254 - sentence2_length
                tokens1 = tokens1[:253 + difference]
        else:
            difference = 254 - sentence1_length
            tokens2 = tokens2[:253 + difference]

    tokens1.append('[SEP]')
    tokens2.append('[SEP]')

    return [tokenizer.convert_tokens_to_ids(tokens1), tokenizer.convert_tokens_to_ids(tokens2)]


def bert_encode(data, tokenizer):
    sentences = data[['sentence1', 'sentence2']].values

    sent1_list = []
    sent2_list = []

    for sent_couple in sentences:
        tokenized_sentences = encode_sentence(tokenizer, sent_couple)
        sent1_list.append(tokenized_sentences[0])
        sent2_list.append(tokenized_sentences[1])

    sentence1 = tf.ragged.constant(sent1_list)
    sentence2 = tf.ragged.constant(sent2_list)

    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])] * sentence1.shape[0]
    input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)

    input_mask = tf.ones_like(input_word_ids).to_tensor()

    type_cls = tf.zeros_like(cls)
    type_s1 = tf.zeros_like(sentence1)
    type_s2 = tf.ones_like(sentence2)
    input_type_ids = tf.concat([type_cls, type_s1, type_s2], axis=-1).to_tensor()

    print(type_cls.shape, type_s1.shape, type_s2.shape, input_type_ids.shape)

    inputs = {
        'input_word_ids': input_word_ids.to_tensor(),
        'input_mask': input_mask,
        'input_type_ids': input_type_ids
    }

    return inputs


def prepare_data():
    train_df = pd.read_csv(config.train_path, sep=',', names=config.header_split, skiprows=1)
    dev_df = pd.read_csv(config.dev_path, sep=',', names=config.header_split, skiprows=1)
    test_df = pd.read_csv(config.test_path, sep=',', names=config.header_split, skiprows=1)

    # Set up tokenizer to generate Tensorflow dataset
    tokenizer = config.tokenizer

    train = bert_encode(train_df, tokenizer)
    train_labels = train_df['score'].astype('int64')

    validation = bert_encode(dev_df, tokenizer)
    validation_labels = dev_df['score'].astype('int64')

    test = bert_encode(test_df, tokenizer)
    test_labels = test_df['score'].astype('int64')

    for key, value in train.items():
        print(f'{key:15s} shape: {value.shape}')

    print(f'train_labels shape: {train_labels.shape}')

    return test, test_labels, train, train_labels, validation, validation_labels


# if __name__ == '__main__':
#    do_datagen(config.input_raw_data, config.whole_dataset, config.cit_lines)
