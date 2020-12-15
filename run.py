import os
import sys
import datagen
import mendley_preparation
import model
import config


def run_model(number, key):
    os.system('head -' + number + ' data/raw-data.csv > data/raw-data_part.csv')
    input_file = config.input_raw_data

    if key.lower() == 'cit':
        output_file = config.whole_dataset_cit
        command = config.command_shuffle_cit
        shuffled = config.whole_dataset_cit_shuf
        other_dataset = config.cit_lines
    else:
        output_file = config.whole_dataset
        command = config.command_shuffle_tags
        shuffled = config.whole_dataset_shuf
        other_dataset = config.tags_lines

    datagen.do_datagen(input_file, output_file, other_dataset)
    os.system(command)
    # datagen.shuffle_and_write(command, shuffled)
    datagen.split_dataset(shuffled, key)
    fitted_bert = model.bert_model(key, other_dataset)
    # print('\n##################################################### testing model...')
    if key.lower() == 'cit':
        mendley_preparation.do_test(fitted_bert)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('USAGE: prompt the desired number of lines, and "CIT" if you want to use the co-citations, or else "TAG" if you want to use tags')
        print("Exiting...")
        sys.exit()

    ignore, number_of_lines, gen_type = sys.argv
    run_model(number_of_lines, gen_type)
