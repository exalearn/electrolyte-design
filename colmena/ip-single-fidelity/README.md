# IP Optimization with Single Fidelity

This application finds the molecule with the largest ionization potential (IP) using active learning. 
It works by first using an ensemble of message passing neural networks (MPNN) to guess the IP of each molecule in a search space and assess an level of uncertainty for each guess.
We feed these predictions and uncertaintities to an experimental design algorithm to prioritize the molecules for which we should compute the IP.
We update the models as IP computations are completed and use the updated models to reprioritize our list of computations.

## Running

This code is designed to be run on Theta at ALCF. The application is configured by passing arguments to `run.sh` via `qsub`, which are then passed to `run.py` along with some hard-coded settings (e.g., specific models to use, optimized batch sizes).

Example: `qsub -n 256 -t 360 run.sh --retrain-frequency 8`

## Implementation

Our application broken into 6 "agent" threads that manage training machine learning models, 
using predictions of models to select the next simulation, 
running simulations,
and allocating resources between the 3 types of computations.

### Model Training

The model training code is managed by two agents.
The `train_models` agent starts working when a `start_training` event is set,
and submits a task to update each MPNN in the ensemble once training data is made available.
The `update_weights` task receives the completed training tasks and uses them to update the weights
of the MPNN object.

### Simulation Selection

The simulation selection task is built using a single agent.
The agent starts when the `start_inference` task is flagged by launching a thread
that submits chunks of molecules to be evaluate with each model.
The rate of task submission by the thread is controlled using a semaphore, `inference_slots`.
The main execution thread collects the results of the inference tasks and uses
the semaphore to signal the submission thread resources are available once it completes.

The results of inference on all models for all molecules are used to prioritize the tasks.
We place the top-ranked tasks in a list, `task_queue`, that is used by the simulation runner.

### Simulation Runner

The simulation runner is composed two agents, submission and receiving.
The `submit_qc` task waits until sufficient nodes are available by watching the thinker's resource counter.
Once nodes are availble, it pulls the next task from the `task_queue` and submits it for execution.

Completed tasks are processed by `process_qc`, which also manages marking nodes from "complete tasks" as complete.
If successful, the results are stored in the database used to retrain the models.


### Resource Allocator

The resource allocator agent varies the amount of resources dedicated to each computational task.

Resources are allocated in several phases:

1. *Initialization*: All resources dedicated to simulation selection. 
1. *Simulation*: All resources dedicated to performing the selected simulations.
1. *Model Updating*: After a specified number of simulations complete, resources are diverted to retraining the model. The agent will reallocate nodes from simulation to training tasks until every training task has started.
1. *Reprioritization*: Once all models are done training, the nodes are re-tasked to inference. The agent continues to reallocate nodes from simulation to inference until the all inference tasks have completed. The agent then returns to the "simulation" phase.


## Change Log

- March 31: Switched to using greedy rather than UCB for the selection criterion
- April 1: Caught bug in model training. Small numbers of model iterations lead to worsening performance
- April 3: Improved the parameters for the model training. Increased learning rate and epochs
- April 8: Changed task selection to not resubmit jobs currently underway to the molecule queue.
