import os
import json
import csv
import config
import datagen
import pandas as pd


def get_titles(biblio):
    citations = []
    for reference in biblio.values():
        try:
            citations.append(reference['title'])
        except TypeError:
            continue

    return citations


def prepare_data(json_files, output_file):

    for file1 in json_files:
        current1 = json.load(open(config.mendley_data_dir + '/' + file1))

        for file2 in json_files:
            current2 = json.load(open(config.mendley_data_dir + '/' + file2))

            try:
                number1 = current1['docId']
                sequence1 = current1['metadata']['title'].lower() + '. ' + current1['abstract'].lower()
                bibs1 = current1['bib_entries']
                papers1 = set(get_titles(bibs1))

                number2 = current2['docId']
                sequence2 = current2['metadata']['title'].lower() + '. ' + current2['abstract'].lower()
                bibs2 = current2['bib_entries']
                papers2 = set(get_titles(bibs2))
            except KeyError:
                continue

            common_cits = len(papers1.intersection(papers2))
            score = datagen.get_score(common_cits, len(papers1), len(papers2))

            with open(output_file, mode='a') as test_file:
                test_writer = csv.writer(test_file, delimiter=',', quotechar='"',
                                         quoting=csv.QUOTE_MINIMAL)

                test_writer.writerow([number1, number2, sequence1, sequence2, score])


def run_bert_on_test_data():
    test_df = pd.read_csv(config.whole_mendley_shuf, sep=',', names=config.header)
    print(test_df.head(10))

    tokenizer = config.tokenizer

    test = datagen.bert_encode(test_df, tokenizer)
    test_labels = test_df['score'].astype('int64')

    return test, test_labels


def do_test(classifier):
    prepare_data(config.last_files, config.whole_mendley)
    os.system(config.command_shuffle_mendley)
    test, labels = run_bert_on_test_data()

    evaluation = classifier.evaluate(x=test,
                                     y=labels,
                                     batch_size=config.batch_size)
    print(evaluation)


if __name__ == '__main__':
    prepare_data(config.last_files, config.whole_mendley)
    os.system(config.command_shuffle_mendley)
    run_bert_on_test_data()
