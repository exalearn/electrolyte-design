#! /bin/bash
search_space=../../notebooks/screen-search-space/runs/ZIN-molwt\=150.0-params\=basic/screened_molecules.csv
#search_space=../../ai-components/search-spaces/MOS-search.csv
search_space_name=ZINC15
# Get the MPNN models

# Relevant endpoints
#  ea5a6ded-ee11-4d0b-9bbd-f33d5f4a3655: Debug queue, on node (for XTB workloads)
#  1d4f13c3-0fbe-486c-b2cc-77c3e4c109db: Debug queue, MPI (for NWChem workloads)
#  acdb2f41-fd86-4bc7-a1e5-e19c12d3350d: Lambda
#  0c1b0c51-2ae5-4401-8b0c-13cdb66c8e47: ThetaGPU single node
#  6c5c793b-2b2a-4075-a48e-d1bd9c5367f6: ThetaGPU full node

python run.py --mongoport 27855 \
       --ml-endpoint acdb2f41-fd86-4bc7-a1e5-e19c12d3350d \
       --qc-endpoint 1d4f13c3-0fbe-486c-b2cc-77c3e4c109db \
       --simulation-spec model-spec.yaml \
       --search-space $search_space \
       --search-space-name $search_space_name \
       --ml-ps-backend globus \
       --simulate-ps-backend globus \
       --simulate-ps-globus-config globus_config-home_theta.json \
       --ml-ps-globus-config globus_config-home_lambda.json \
       --molecules-per-ml-task 250000 \
       $@
