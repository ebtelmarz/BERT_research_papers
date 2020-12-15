import os
from official.nlp import bert
import official.nlp.bert.tokenization

################################################################################### MODEL
bert_folder = 'uncased_L-12_H-768_A-12'
epochs = 3
batch_size = 8          # low cause of OOM errors
eval_batch_size = 32

################################################################################### DATASET
command_shuffle_cit = 'shuf intermediate/dataset_cit.csv -o shuffled/shuf_dataset_cit.csv'
command_shuffle_tags = 'shuf intermediate/dataset.csv -o shuffled/shuf_dataset.csv'
command_shuffle_mendley = 'shuf intermediate/mendley_papers.csv -o shuffled/shuf_mendley_papers.csv'
command_remove_app = 'rm shuffled/app.csv'

intermediate_data_dir = 'intermediate'
shuffled_dir = 'shuffled'
data_dir = 'data'
mendley_data_dir = 'json'
export_dir = 'saved_model'

input_raw_data = 'raw-data_part.csv'
whole_dataset = os.path.join(intermediate_data_dir, 'dataset.csv')
whole_dataset_shuf = os.path.join(shuffled_dir, 'shuf_dataset.csv')

whole_dataset_cit = os.path.join(intermediate_data_dir, 'dataset_cit.csv')
whole_dataset_cit_shuf = os.path.join(shuffled_dir, 'shuf_dataset_cit.csv')

whole_mendley = os.path.join(intermediate_data_dir, 'mendley_papers.csv')
whole_mendley_shuf = os.path.join(shuffled_dir, 'shuf_mendley_papers.csv')

input_test_raw_data = 'raw-data_tail.csv'
test_whole_dataset = 'test_data.csv'
test_whole_dataset_cit = 'test_data_cit.csv'

train_path = 'dataset/train.csv'
dev_path = 'dataset/dev.csv'
test_path = 'dataset/test.csv'

train_path_cit = 'dataset/train_cit.csv'
dev_path_cit = 'dataset/dev_cit.csv'
test_path_cit = 'dataset/test_cit.csv'

tags_lines = open(os.path.join(data_dir, 'item-tag.dat')).readlines()
cit_lines = open(os.path.join(data_dir, 'citations.dat')).readlines()

header = ['id1', 'id2', 'sentence1', 'sentence2', 'score']
header_split = ['', 'id1', 'id2', 'sentence1', 'sentence2', 'score', 'split', 'split_1']

tokenizer = bert.tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_folder, "vocab.txt"),
        do_lower_case=True)
