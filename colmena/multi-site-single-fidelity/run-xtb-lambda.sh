#! /bin/bash
search_space=../../notebooks/screen-search-space/runs/ZIN-molwt\=200.0-params\=basic/screened_molecules.csv
#search_space=../../ai-components/search-spaces/MOS-search.csv
search_space_name=ZINC15
# Get the MPNN models
mpnn_dir=../../ai-components/ip-multi-fidelity/ip-vacuum-xtb/smiles
mpnn_files=$(find $mpnn_dir -name best_model.h5 | sort | tail -n 8)

# Relevant endpoints
#  ea5a6ded-ee11-4d0b-9bbd-f33d5f4a3655: Debug queue, on node (for XTB workloads)
#  acdb2f41-fd86-4bc7-a1e5-e19c12d3350d: Lambda
#  0c1b0c51-2ae5-4401-8b0c-13cdb66c8e47: ThetaGPU single node
#  6c5c793b-2b2a-4075-a48e-d1bd9c5367f6: ThetaGPU full node

python run.py --mongoport 27855 \
       --ml-endpoint acdb2f41-fd86-4bc7-a1e5-e19c12d3350d \
       --qc-endpoint ea5a6ded-ee11-4d0b-9bbd-f33d5f4a3655 \
       --qc-specification xtb \
       --mpnn-model-files $mpnn_files \
       --search-space $search_space \
       --search-space-name $search_space_name \
       --infer-ps-backend globus \
       --train-ps-backend globus \
       --simulate-ps-backend file \
       --ps-file-dir proxy-store-scratch \
       --ps-globus-config globus_config.json \
       --molecules-per-ml-task 250000 \
       $@
