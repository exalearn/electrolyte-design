#! /bin/bash
# Run the design app on a desktop computer
export CUDA_VISIBLE_DEVICES=

# Get the MPNN models
model_dir=../../ai-components/hpdc-2022/xtb-models/
mpnn_dir=$model_dir/networks
mpnn_files=$(find $mpnn_dir -name best_model.h5 | sort | tail -n 8)


python run.py --ml-endpoint cc63787b-7425-4eee-8b39-d34a23a9d2e6 \
       --qc-endpoint cc63787b-7425-4eee-8b39-d34a23a9d2e6 \
       --qc-specification xtb \
       --mpnn-model-files $mpnn_files \
       --training-set $model_dir/records.json \
       --search-space $model_dir/../QM9-search.tsv \
       $@
