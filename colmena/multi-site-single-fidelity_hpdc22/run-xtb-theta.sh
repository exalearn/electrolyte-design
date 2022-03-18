#! /bin/bash
# Run the design app on a desktop computer
export CUDA_VISIBLE_DEVICES=

# Get the MPNN models
model_dir=../../ai-components/hpdc-2022/xtb-models/
mpnn_dir=$model_dir/networks
mpnn_files=$(find $mpnn_dir -name best_model.h5 | sort | tail -n 2)


python run.py --ml-endpoint 0c1b0c51-2ae5-4401-8b0c-13cdb66c8e47 \
       --qc-endpoint cc63787b-7425-4eee-8b39-d34a23a9d2e6 \
       --qc-specification xtb \
       --mpnn-model-files $mpnn_files \
       --training-set $model_dir/records.json \
       --search-space $model_dir/../QM9-search.tsv \
       --infer-ps-backend globus \
       --train-ps-backend globus \
       --simulate-ps-backend redis \
       --ps-globus-config globus-configs/logan-desktop.json \
       $@
