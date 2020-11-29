import os
from official.nlp import bert
import official.nlp.bert.tokenization

################################################################################### MODEL
bert_folder = 'uncased_L-12_H-768_A-12'
epochs = 3
batch_size = 32
eval_batch_size = 32

################################################################################### DATASET
data_dir = 'data'
whole_dataset = 'dataset.csv'
whole_dataset_cit = 'dataset_cit.csv'
input_raw_data = 'raw-data_part.csv'

train_path = 'dataset/train.csv'
dev_path = 'dataset/dev.csv'
test_path = 'dataset/test.csv'

tags_lines = open(os.path.join(data_dir, 'item-tag.dat')).readlines()
cit_lines = open(os.path.join(data_dir, 'citations.dat')).readlines()

header = ['id1', 'id2', 'sentence1', 'sentence2', 'score']

tokenizer = bert.tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_folder, "vocab.txt"),
        do_lower_case=True)