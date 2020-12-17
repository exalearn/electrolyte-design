#! /bin/bash
# Script for running the code with the XTB components, useful for debugging/dev work

# Define the version of models to use
mpnn_dir=../../ai-components/mpnn/nwchem-atomization-v0/
search_space=../../ai-components/search-spaces/G13-filtered.csv


models=`find $mpnn_dir -name best_model.h5 | sort | head -n 1`
python run.py --mpnn-config-directory $mpnn_dir \
    --mpnn-model-files $models \
    --search-space $search_space \
    --qc-parallelism 2 \
    --qc-spec small_basis $@
