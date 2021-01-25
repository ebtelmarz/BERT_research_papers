import os
import sys
import datagen
import mendley_preparation
import model
import config


def run_model(number):
    os.system('head -' + number + ' data/raw-data.csv > data/raw-data_part.csv')

    datagen.do_datagen(config.input_raw_data, config.whole_dataset, config.cit_lines)
    mendley_preparation.prepare_data(config.last_files, config.whole_dataset)
    os.system(config.command_shuffle)
    datagen.split_dataset(config.whole_dataset_shuf)

    fitted_bert = model.bert_model()

    mendley_preparation.do_test(fitted_bert)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('USAGE: prompt the desired number of lines')
        print("Exiting...")
        sys.exit()

    ignore, number_of_lines = sys.argv
    run_model(number_of_lines)
