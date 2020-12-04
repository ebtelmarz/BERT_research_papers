# BERT NLP for research papers

This repo is an attempt to use BERT as a classifier to compute semantic similarity between pairs of research papers represented by their title and abstract concatenated, labeling them as similar or not similar.

## References
- [Official Google BERT repo](https://github.com/google-research/bert)
- Paper: [Tag-Aware Document Representation for Research Paper Recommendation](https://www.researchgate.net/publication/343319230_Tag-Aware_Document_Representation_for_Research_Paper_Recommendation)
- Paper: [BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
- Paper: [Aspect-based Document Similarity for Research Papers](https://arxiv.org/pdf/2010.06395.pdf)


## Environment
- Ubuntu 18.04
- Python 3.6.9

## Usage
clone or download this repository
```bash
git clone https://github.com/ebtelmarz/BERT_research_papers.git
```
move inside the downloaded folder
```bash
cd BERT_research_papers
```
execute the run.sh file by running  
```bash
sh run.sh
```

### run.sh
running the run.sh file will execute the following commands:

- install all python3 requirements
```bash
pip3 install -r requirements.txt
```

- download the Bert model from the official repo of google, unzip it and then remove the zip file
```bash
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

unzip uncased_L-12_H-768_A-12.zip

rm uncased_L-12_H-768_A-12.zip
```
- download the [citeulike-a](https://github.com/js05212/citeulike-a) dataset from the official repo and move it to the data directory
```bash
git clone https://github.com/js05212/citeulike-a.git

mv citeulike-a/ data/
```
- copy the first [N] lines of the dataset to a new file, you can specify the desired [N] value, the default is 501
```bash
head -[N] data/raw-data.csv > data/raw-data_part.csv
```
- execute the code
```bash
python3 run.py
```
