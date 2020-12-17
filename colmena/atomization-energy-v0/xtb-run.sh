#! /bin/bash
# Script for running the code with the XTB components, useful for debugging/dev work

# Define the version of models to use
moldqn_dir=../../ai-components/moldqn/xtb-atomization-v1/
mpnn_dir=../../ai-components/mpnn/xtb-atomization-v1/


models=`find $mpnn_dir -name best_model.h5 | sort | head -n 4`

python run.py --mpnn-config-directory $mpnn_dir \
    --mpnn-model-files $models \
    --initial-agent $moldqn_dir/agent.pkl \
    --initial-search-space $moldqn_dir/best_mols.json \
    --initial-database $mpnn_dir/initial_database.json \
    --qc-parallelism 16 \
    --qc-spec xtb $@
