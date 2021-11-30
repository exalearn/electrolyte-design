#! /bin/bash
#COBALT -A CSC249ADCD08 --attrs enable_ssh=1

# Load up the Python environment
module load miniconda-3/latest
conda activate ../../env

# Add NWChem and Psi4 to the path
export PATH=~/software/psi4/bin:$PATH

# Start the redis-server
port=631${RANDOM::2}
redis-server --port $port &> redis.out &
redis=$!

# Run!
# Define the version of models to use
prod_model_dir=../../ai-components/ip-multi-fidelity/ea-vacuum-dfb-adia-smb/
mpnn_dir=$prod_model_dir/vertical
vertical=$(find $mpnn_dir -name "*best_model.h5" | sort | tail -n 8)

adiabatic_dir=$prod_model_dir/adiabatic
adiabatic=$(find $adiabatic_dir -name "*best_model" | sort | tail -n 8)

normal_dir=$prod_model_dir/normal
normal=$(find $normal_dir -name "*best_model" | sort | tail -n 8)

# Set the search space
search_space=../../ai-components/search-spaces/pthalimide.tsv

# Set the Python path to include sim.py in this directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run!
# Calibration factors are determined in notebooks/production-models/**/calibration-results.csv
python run-multi.py --redisport $port --mongo-url mongo://thetalogin6:27845/ \
    --vertical-model-files $vertical \
    --vertical-calibration 1.82 \
    --adiabatic-model-files $adiabatic \
    --adiabatic-calibration 1.80 \
    --normal-model-files $normal \
    --normal-calibration 1.82 \
    --search-space $search_space \
    --max-heavy-atoms 25 \
    --target-level dfb-vacuum-smb-geom \
    --target-range -1000 1 \
    --molecules-per-ml-task 4096 \
    --train-timeout 900 \
    --num-qc-nodes 164 \
    $@

# Kill the servers
kill $redis
