#!/bin/bash

resDir="./results/"   # Directory name to save the results
train_x="train_X.csv" # Training dataset filename
train_y="train_Y.csv" # Subtype label filename for training dataset
test_x="test_X.csv"   # Testing dataset filename
test_y="test_Y.csv"   # Subtype label filename for testing dataset
num_gene="2000"        # Number of genes (from prepare_data feature_counts.txt)
num_cpg="2000"         # Number of CpG clusters (from prepare_data feature_counts.txt)
num_mirna="200"       # Number of microRNAs (from prepare_data feature_counts.txt)

mkdir -p "$resDir"
python moBRCA-net.py $train_x $train_y $test_x $test_y $num_gene $num_cpg $num_mirna $resDir

