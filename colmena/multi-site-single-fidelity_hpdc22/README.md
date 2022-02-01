# Multi-Site Active Learning for IP Optimization

Re-training machine learning models based on new simulations is major bottleneck in our [previous molecular design application](../ip-single-fidelity/README.md)
and one we can reduce by performing training on specialized hardware.
Here, we illustrate how to implement such an application using Colmena backed by FuncX to distribute each type of task to specialized resources.

## Running the Code

The campaign driver, `run.py`, defines a series of command line arguments that control how the campaign is run.

One key option, `--qc-specification`, defines which simulation code to use for the quantum chemistry. of the search that have different hardware requirements for the simulation code. 
One uses HPC NWChem that requires prodigious HPC resource 
and a second that uses XTB that can be run on a single compute note. 

The choice of QC code means you should select different sets of ML models (`--mpnn-model-files`) and their corresponding training sets, among other options.
You will also need to specify an appropriate FuncX endpoint on which to run the computations (``--qc-endpoint``).
More details about configuring the FuncX endpoint are below.

Further details of the search algorithm is controlled by "Problem Definition" and "Model Training" settings.
Such settings include the number of molecules sampled, how often to retrain the model, and related settings.

Call `python run.py --help` for full details.

## Configuring the Endpoint

**TODO**: Write about how to set up FuncX

## ProxyStore

ProxyStore can be enabled on a per-topic basis (e.g., for the simulate, infer, and train tasks) using the provided command line arguments.
See the ProxyStore argument group with `run.py --help`.

### Example ProxyStore Usage

In this example, `run.py` is executed on a Theta login node, a Redis server is running on a `thetamom1`, simulations are done on a Theta endpoint `THETA_ENDPOINT`, and inference and training tasks are done on a ThetaGPU endpoint `THETAGPU_ENDPOINT`.

```
$ run.py \
      --redishost thetamom1 \
      --redisport $REDIS_PORT \
      --qc-endpoint $THETA_ENDPOINT \
      --ml-endpoint $THETAGPU_ENDPOINT \
      --simulate-ps-backend redis \
      --infer-ps-backend file \
      --train-ps-backend file \
      --ps-threshold 500000 \
      --ps-file-dir $PROJECT_DIR/scratch/proxystore-dump
```

With the above configuration, with `simulate` tasks, ProxyStore with Redis will be used (will default to use the same Redis server that the Task server uses).
When using the Redis ProxyStore backend, the Redis server must be reachable from the Colmena client and workers on the FuncX endpoint.
This is why we place the Redis server on a Theta MOM node in this example.

For the `infer` and `train` tasks, we use a file system backend with ProxyStore.
This is because workers on the ThetaGPU FuncX endpoint cannot access our Redis server running on the Theta MOM node but can access the Theta file system.
The `--ps-file-dir` argument specifies a directory that ProxyStore can use for storing serialized objects.

For all ProxyStore backends, only objects greater than 500KB will be proxied (as specified by the `--ps-threshold 500000` argument).

`globus` is a third ProxyStore backend that is supported in addition to `redis` and `file`.
When using the `globus` backend option, a ProxyStore Globus config file must also be specified via `--ps-globus-config`.
An example is provided in `globus_config.json`.

