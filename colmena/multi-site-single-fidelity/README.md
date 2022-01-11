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

**TODO**: Add in details about configuring ProxyStore
