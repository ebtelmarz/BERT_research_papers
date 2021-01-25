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

wget https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/zm33cdndxs-2.zip
unzip zm33cdndxs-2.zip
unzip json-articals.zip

rm zm33cdndxs-2.zip
rm json_scheme.txt
rm LICENCE.md
rm change_log.txt
rm README.md
rm os-ccby-40k-ids.csv
rm ELSEVIERCC_BYCORPUS.pdf
rm json-articals.zip

mkdir dataset
mkdir shuffled
mkdir intermediate