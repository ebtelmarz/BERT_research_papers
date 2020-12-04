#!/usr/bin/env bash

echo 'installing python3 packages...'
pip3 install -r requirements.txt

echo 'downloading bert-model...'
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
rm uncased_L-12_H-768_A-12.zip

echo 'downloading dataset...'
git clone https://github.com/js05212/citeulike-a.git
mv citeulike-a/ data/

head -501 data/raw-data.csv > data/raw-data_part.csv

echo 'running...'
python3 run.py