# Molecular Optimization for Atomization

This application is designed to find molecules with minimal atomization energies
using a combination of DFT, message-passing neural networks and reinforcement-learning-based
molecular generation schemes. 

## Approach

We use a two-level optimization scheme.

Level 1 is a Reinforcement Learning algorithm that attempts to find molecules with 
minimal atomization energy estimated by a MPNN model.
The MPNN operates using only the graph structure for the molecule,
which allows it to make predictions very quickly and simplifies the
RL agent by only considering molecular bonding and not 3D conformer structure.

Level 2 is an Active Learning optimization where we find the molecules
suggested by Level 1 which have the lowest atomization energies.
Here, we use quantum chemistry to compute the atomization energy 
rather than relying on an MPNN to estimate it.
The active learning algorithm uses predictions from the MPNNs used
in Level 1 to identify which quantum chemistry calculations are the most valuable.
The new quantum chemistry calculations are used to re-train the MPNN, 
which is sent back to the "Level 1" optimizer.

## Running this Application

First, start a Redis server on your system (`redis-server`).

Then, modify `run.py` to define a Parsl configuration appropriate for your system.
The application is currently configured to run on Theta. 
(Documentation pending). 

Launch the code by calling `python run.py`.
The code is built using `argparse`, so calling `python run.py --help` will describe available options.

The outcome of the run will be stored in a subdirectory of the `runs` folder and named 
using the launch time of the application.

See [`run.sh`](./run.sh) as an example run script that combines each of these steps,
and [`nwchem-run.sh`](./nwchem-run.sh) to see common arguments for a run.

## Implementation

Our application is composed into four independent Colmena agents that share a:

- *Result Database*: A collection of molecules and their computer properties 
- *Model Library*: A collection of message passing neural networks (MPNNs) trained on the Result Database
- *Task Queue*: A list of molecules, prioritized by 
- *Search Space*: A list of molecules that could be simulated

### Simulation Runner

Submits new quantum chemistry tasks and records results to the Result Database.
Initially submits as many tasks as available workers and then waits on 
a Colmena queue for a simulation to finish.
Each time a task completes, the agent stores the result in a database, 
submits the next molecule in the Task Queue and removes the submitted
molecule from the Search Space.

### Model Updater

Continually updates the Model Library based on the latest results from the simulation runner.
Has a first-in-first-out queue of models to be updated. 
When a model finishes re-training, it is added to the queue and the model which is
at the front of the queue is send to re-train for a few epochs.

### Molecule Generator

Runs the MolDQN to generate new molecules and gradually train MolDQN's policy network.
Randomly selects using a randomly-selected MPNN from the Model Library 
to use as an reward function before beginning the computation.
The agent then waits on the Colmena redis queue until the MolDQN finishes 
a few training episodes and then returns an updated agent with the list of molecules
that were sampled.
The agent will store these generated molecules in the Search Space before
re-submitting the agent.

### Task Ranker

Continually re-prioritizes the Task Queue.
Iteratively runs inference of all molecules in the Search Space using every model in 
the Model Library. 
Once the inference completes, the agent employs active learning algorithms to rank
 molecules based on  the mean and disagreement between the Model Library models.
The agent resets and replaces the Task Queue with molecules based on these new 
rankings before repeating.
