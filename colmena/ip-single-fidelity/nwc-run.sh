#! /bin/bash
# Script for running the code with the XTB components, useful for debugging/dev work

# Define the version of models to use
mpnn_dir=/lus/theta-fs0/projects/CSC249ADCD08/edw/ai-components/sc-2021/mpnn
search_space=/lus/theta-fs0/projects/CSC249ADCD08/edw/ai-components/search-spaces/QM9-search.tsv


models=`find $mpnn_dir/smb-ip -name best_model.h5 | sort | head -n 4`
python run.py --mpnn-config-directory $mpnn_dir \
    --mpnn-model-files $models \
    --search-space $search_space \
    --nodes-per-task 2 \
    --retrain-frequency 1 \
    $@
