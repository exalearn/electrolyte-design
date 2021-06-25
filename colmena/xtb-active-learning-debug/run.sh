#! /bin/bash
#COBALT -A CSC249ADCD08 --attrs enable_ssh=1

mpnn_dir=../../ai-components/sc-2021/mpnn
# Define the version of models to use
search_space=../../ai-components/search-spaces/QM9-search.tsv
models=`find $mpnn_dir/xtb-ip -name best_model.h5 | sort | head -n 2`

export PYTHONPATH=$PYTHONPATH:`pwd`

# Run!
python run.py --mpnn-config-directory $mpnn_dir \
    --mpnn-model-files $models \
    --init-dataset full_dataset.json \
    --search-space $search_space \
    $@
