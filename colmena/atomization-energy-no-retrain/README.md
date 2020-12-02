# Single-Pass ML Atomization Energy

This application is designed to find molecules with minimal atomization energies
using a gredy ML-based search for new molecules.

## Approach

The alogirthm here is simple: infer properties of a large number of potential molecules 
with a machine model and then run quantum chemistry of those that have the best properties.

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
and [`nwc-run.sh`](./nwchem-run.sh) to see common arguments for a run.

## Implementation

Our application is composed into four Colmena agents.

The method server only runs two types of methods "ML inference" and "QC simulation."
The QC simulations are not run until after ML inference tasks complete,
so we can re-use the same computational resources for both types of tasks.
Once ML tasks are completed, Parsl will reconfigure the system to run QC tasks.

### Search Space Reader

Maintains a queue of batches of molecules to be evaluated with machine learning.

Runs a loop that reads a batch of molecules, downsamples them if we are only running
a fraction of the search space, and writes the batch to an output queue.
The queue is ensures that the ML Task Submitter always has a batch of 
molecules ready to submit when a previous batch is completed.

### ML Task Submitter

Submits batches of molecules to be evaluated with a machine learning models.

Reads from the queue of molecules batches from the Search Space Reader
and submits them to Colmena. 

Uses a [semaphore](https://docs.python.org/3/library/threading.html#threading.Semaphore)
to ensure too many jobs are not submitted to Colmena. 
We control the number of active jobs to reduce memory usage 
(e.g., not reading 1B molecules into memory at the same time)
and to enable better overhead monitoring because
we cannot yet distinguish how long a job is waiting for use.

### ML Task Consumer / QC Task Submitter

Uses inference results to determine the top molecules,
starts quantum chemistry calculations for top molecules.

Each time a ML inference task completes, 
this agent updates a list of top molecules and
releases resources from the semaphore shared with the 
ML task submitter.

Once all ML tasks have completed, submits
quantum chemistry tasks.
We use a similar semaphore-based strategy to throttle
the number of tasks being submitted at any one time.

### QC Task Consumer

Retrieves completed tasks from Colmena result queue, 
writes results to disk and marks resources as free
for the QC Task Submitter.
