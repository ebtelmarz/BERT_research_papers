import os
from official.nlp import bert
import official.nlp.bert.tokenization

################################################################################### MODEL
bert_folder = 'uncased_L-12_H-768_A-12'
epochs = 3
batch_size = 8          # low cause of OOM errors
eval_batch_size = 32

################################################################################### DATASET
command_shuffle = 'shuf intermediate/dataset.csv -o shuffled/shuf_dataset.csv'
command_shuffle_mendley = 'shuf intermediate/mendley_papers.csv -o shuffled/shuf_mendley_papers.csv'

intermediate_data_dir = 'intermediate'
shuffled_dir = 'shuffled'
data_dir = 'data'
# mendley_data_dir = 'json'
export_dir = 'saved_model'
new_mendley_dir = 'mendley'

# json_files = os.listdir(mendley_data_dir)
# top_files = json_files[:100]
# last_files = json_files[-200:]
# json_titles = os.path.join(new_mendley_dir, 'mendley_papers_list.csv')
# json_titles_citations = os.path.join(new_mendley_dir, 'mendley_citations.csv')

input_raw_data = 'raw-data_part.csv'

zeros_file = os.path.join(intermediate_data_dir, 'zeros.csv')

whole_dataset = os.path.join(intermediate_data_dir, 'dataset.csv')
intermediate_dataset = os.path.join(intermediate_data_dir, 'intermediate_dataset.csv')
whole_dataset_shuf = os.path.join(shuffled_dir, 'shuf_dataset.csv')

whole_mendley = os.path.join(intermediate_data_dir, 'mendley_papers.csv')
whole_mendley_shuf = os.path.join(shuffled_dir, 'shuf_mendley_papers.csv')

test_whole_dataset = 'test_data.csv'

train_path = 'dataset/train.csv'
dev_path = 'dataset/dev.csv'
test_path = 'dataset/test.csv'

tag_lines = open(os.path.join(data_dir, 'item-tag.dat')).readlines()
# cit_lines = open(os.path.join(data_dir, 'citations.dat')).readlines()
tags_number = len(open(os.path.join(data_dir, 'tags.dat')).readlines())

header = ['id1', 'id2', 'sentence1', 'sentence2', 'score']
header_split = ['', 'id1', 'id2', 'sentence1', 'sentence2', 'score', 'split', 'split_1']

# class_labels = ['Not Similar', 'Mild  Similar', 'Similar']
class_labels = ['Not Similar', 'Similar']

tokenizer = bert.tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_folder, "vocab.txt"),
        do_lower_case=True)
