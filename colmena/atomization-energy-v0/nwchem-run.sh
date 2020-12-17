#! /bin/bash
# Script for running the code with the XTB components, useful for debugging/dev work

# Define the version of models to use
moldqn_dir=../../ai-components/moldqn/nwchem-atomization-v0/
mpnn_dir=../../ai-components/mpnn/nwchem-atomization-v0/


models=`find $mpnn_dir -name best_model.h5 | sort | head -n 4`

python run.py --mpnn-config-directory $mpnn_dir \
    --mpnn-model-files $models \
    --initial-agent $moldqn_dir/agent.pkl \
    --initial-search-space $moldqn_dir/best_mols.json \
    --initial-database $mpnn_dir/initial_database.json \
    --parallel-updating 1 \
    --qc-spec small_basis $@
