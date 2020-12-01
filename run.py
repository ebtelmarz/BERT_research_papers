import os
import datagen
import model


def run_model():
    # only for dev purp
    os.system('head -101 data/raw-data.csv > data/raw-data_part.csv')
    print('##################################################### generating raw dataset...')
    datagen.tag_based_datagen()
    os.system('shuf dataset.csv -o shuf_dataset.csv')

    # citation_based_datagen()
    # os.system('shuf dataset_cit.csv -o shuf_dataset_cit.csv')
    print('##################################################### splitting dataset...')
    datagen.split_dataset()
    model.bert_model()


if __name__ == '__main__':
    run_model()
