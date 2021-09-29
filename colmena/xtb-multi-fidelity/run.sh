#! /bin/bash
#COBALT -A CSC249ADCD08 --attrs enable_ssh=1

# Define the version of models to use
mpnn_dir=../../notebooks/production-models/ip-vacuum_xtb/mpnn/bootstrap-ensemble/networks/
mpnns=$(find $mpnn_dir -name best_model.h5 | sort | tail -n 8)

schnet_dir=../../notebooks/production-models/ip-vacuum_xtb/schnet-delta_xtb-vertical/bootstrap-ensemble/networks/
schnets=$(find $schnet_dir -name best_model | sort | tail -n 8)

# Set the search space
search_space=../../ai-components/search-spaces/QM9-search.tsv

# Set the Python path to include sim.py in this directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Make a fresh copy of the database
cp -r ../../db .
mongod --dbpath ./db > mongo.log &
mongo_pid=$!

# Run!
# Calibration factors are determined in notebooks/production-models/**/calibration-results.csv
python run.py --mpnn-config-directory $mpnn_dir/../../.. \
    --mpnn-model-files $mpnns \
    --mpnn-calibration 1.64 \
    --schnet-model-files $schnets \
    --schnet-calibration 2.21 \
    --search-space $search_space \
    $@

# Stop and eliminate the database
kill $mongo_pid
rm -r ./db

