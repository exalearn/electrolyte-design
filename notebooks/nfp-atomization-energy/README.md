# MPNN Training

Scripts for training a baseline MPNN using QM9 data and the [nfp](https://github.com/NREL/nfp) package.

We provide a testing script that allows you to change basic parameters of an architecture that relies on 
a global state variable. You can also turn on a "maximum atom count" for the data loader (`--padded-size`)
to ensure all batches have the same input shape.
