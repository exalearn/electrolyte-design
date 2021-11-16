# IP Optimization with Multiple Fidelity using ALCF Theta

This application finds the molecule with the largest ionization potential (IP) using multi-fidelity active learning.
Our application computes the IP for a molecule in three steps: first computing the vertical IP and then the adiabatic IP for the molecule with a small basis set (3-21g),
and then re-evaluating molecular energies with a larger basis set (6-31g(2df,p)).
We decide which computation to run at what level based on machine learning models that take
any information we already have about a molecule (e.g., vertical IP) to infer its properties at the highest level of fidelity.
These "simulation" and "decision" policies run concurrently. 
Simulation tasks are run in the order of suggested by the decision process;
the decision process is rerun as more data is acquired during a simulation.

*Simulation* tasks are performed using NWChem driven through its 
[QCEngine](https://github.com/MolSSI/QCEngine) interface.
We run three different types of tasks:
1. Vertical IP with B3LYP/3-21g: A geometry optimization of the neutral molecule then an energy evaluation in the charged state.
1. Adiabatic IP with B3LYP/3-21g: A geometry optimization of the charged molecule
1. Energy with B3LYP/6-31g(2df,p): Energy computation with a larger basis set. Will run energies for the neutral and charged geometries
   with and without solvent, if desired.

*Machine Learning* tasks use two different classes of machine learning models.
Message-passing neural networks (MPNNs) predict molecular properties from the graph structure and are written using TensorFlow.
We use MPNNs for predicting the IP for molecules before any simulations are run.
Continuous-convolution neural networks (i.e., SchNet) predict properties from the 3D geometry of a molecule, 
and are written using [SchNetPack](https://schnetpack.readthedocs.io/en/stable/) and PyTorch.
We use SchNet for predicting a correction factor between the IP computed with lower levels of theory and the target elvel.

## Running the Code

This code is designed to be run on a single workstation and requires launching three processes:
- Redis 
- The python steering process: `python run.py` 
- MongoDB: Running

The `./run.sh` script launches Redis and the Python code, contains the connection information for MongoDB, 
and has the path to the ML models hard-coded.

Call `python run.py --help` to see the full available options for the steering process.ing

## Implementation

Our application broken into 3 pairs of Colmena "agent" threads that manage training machine learning models, 
using predictions of models to select the next simulation,
and running simulations.
Each thread starts when certain conditions are met (e.g., resources coming available, a result being received).
We describe the triggers for each pair of events along with which computations they complete in the following sections.

At present, we will allocate all available nodes to inference tasks and up to the number of models (typically 24) nodes to training.

### Model Training

**Trigger**: A "start training" event is set when a certain number of "high fidelity" computations have completed

**Resources**: Uses resources from the "training" resource pool, which are drawn from the "simulation" pool and 
then dispersed to the "inference" pool after training completes.

The *Train Models* agent gathers the training set for each model and then submits training tasks to the Task Server
as resources are available. 
Datasets are assembled by querying MongoDB for results with
Both classes of models, MPNN and SchNet, use the adaiabtic IPs to train the model but differ in the information used
in the training set.
The MPNNs are provided with the graph of the molecule's bonds and the predict the adiabatic IP,
SchNet models are given the 3D geometry and predict the difference between the vertical and adiabatic IPs.

The *Update Models* agent waits to receive training results and then saves the updated model to disk.

Both agents trigger the "start inference" event and terminate when all models have been retrained.

### Simulation Selection

**Trigger**: The "start inference" event is triggered at the beginning of a run or when a certain number of computations have completed.

**Resources**: Uses resources from the "simulation" resource pool, which are drawn from the "simulation" pool and 
then dispersed back to the "simulation" pool after training completes.

The *Launch Inference* agent gathers the list of molecules to be run,
associates each molecule task with the simulation method to be run if selected,
prepares storage for the inference results,
and then submits the inference tasks.
Molecules to be evaluated are selected two ways:
   1. Identifying molecules from the user-provided search space that have yet to be simulated. These are candidates for computing the vertical IPs, and will be evaluated using the MPNNs.
   2. Querying MongoDB for molecules sum but not all of the simulations. These are candidates for computing adiabatic IPs or IPs with larger basis sets, and will be evaluated with the SchNet models.

If the models have not been retrained since the simulation selection was updated last, we only run the molecules 
for which new simulation data has been acquired. 

The *Record Inference* agent waits until the storage for the inference results is complete
and then stores the results of the inference (predictions of molecular properties) into the pre-allocated arrays.
Once all inference tasks are recorded, it computes the predicted IP and uncertainty in the prediction for all molecules
and uses that information to prepare a **task queue** of molecules to evaluate.
The task queue is a priority queue ordered by the score assigned to each molecule.
If the models have been re-trained since the last time the inference was started,
all current entires are deleted from the task queue.
Each entry in the task list contains identity of the molecule (i.e., an InChI string),
the name of the method to be run for that molecule,
and some debugging information (e.g., the predicted IP and the prediction uncertainty).

### Simulation Runner

The simulation runner is composed two agents, submission and recorder, that each run using different triggers.

#### Simulation Submitter

**Trigger**: Resources available in the "simulation" resource pool

**Resources**: The "simulation" pool. Does not allocate its own resources

This task pops the net simulation from the **task queue** (see [simulation selection](#simulation-selection)),
adds the molecule to the list of simulations that have been run,
and then submits it to the Task Server.

If the task is to compute a larger basis set, it submits energy evaluations separately so they run in parallel.

#### Simulation Recorder

**Trigger**: A simulation task completes

**Resources**: N/A. This agent submits no new tasks.

The simulation recorder performs a several steps whenever an XTB task completes:

- Release the hold on a resource from the simulation pool
- Update the record for the molecule in the MongoDB
- Check if enough highest-fidelity simulations have completed to retrain a model. If so, trigger retraining 
- Check if enough low-fidelity simulations have completed to re-run inference. If so, trigger inference
- Write the QCEngine records of the computation to disk
