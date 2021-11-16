#! /bin/bash
#COBALT -A CSC249ADCD08 --attrs enable_ssh=1

# Load up the Python environment
module load miniconda-3/latest
conda activate ../../env

# Add NWChem and Psi4 to the path
export PATH=~/software/psi4/bin:$PATH
export OMP_NUM_THREADS=64
export KMP_INIT_AT_FORK=FALSE

export PATH="/lus/theta-fs0/projects/CSC249ADCD08/software/nwchem-6.8.1/bin/LINUX64:$PATH"
mkdir -p scratch  # For the NWChem tasks
which nwchem
hostname
module load atp
export MPICH_GNI_MAX_EAGER_MSG_SIZE=16384
export MPICH_GNI_MAX_VSHORT_MSG_SIZE=10000
export MPICH_GNI_MAX_EAGER_MSG_SIZE=131072
export MPICH_GNI_NUM_BUFS=300
export MPICH_GNI_NDREG_MAXSIZE=16777216
export MPICH_GNI_MBOX_PLACEMENT=nic
export MPICH_GNI_LMT_PATH=disabled
export COMEX_MAX_NB_OUTSTANDING=6
export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64_lin/:/opt/intel/compilers_and_libraries_2020.0.166/linux/compiler/lib/intel64_lin:$LD_LIBRARY_PATH

# Start the redis-server
port=631${RANDOM::2}
redis-server --port $port &> redis.out &
redis=$!

# Run!
# Define the version of models to use
prod_model_dir=../../ai-components/ip-multi-fidelity/ip-acn-nob-adia-smb/
mpnn_dir=$prod_model_dir/vertical
vertical=$(find $mpnn_dir -name "*best_model.h5" | sort | tail -n 8)

adiabatic_dir=$prod_model_dir/adiabatic
adiabatic=$(find $adiabatic_dir -name "*best_model" | sort | tail -n 8)

normal_dir=$prod_model_dir/normal
normal=$(find $normal_dir -name "*best_model" | sort | tail -n 8)

# Set the search space
search_space=../../ai-components/search-spaces/QM9-search.tsv

# Set the Python path to include sim.py in this directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run!
# Calibration factors are determined in notebooks/production-models/**/calibration-results.csv
python run.py --redisport $port --mongo-url mongo://thetalogin6:27845/ \
    --vertical-model-files $vertical \
    --vertical-calibration 1.85 \
    --adiabatic-model-files $adiabatic \
    --adiabatic-calibration 1.63 \
    --normal-model-files $normal \
    --normal-calibration 0.99 \
    --search-space $search_space \
    --target-level nob-acn-smb-geom \
    --oxidize \
    $@

# Kill the servers
kill $redis
