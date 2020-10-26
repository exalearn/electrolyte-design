# Electrolyte Design Toolkit

Design toolkit for molecular electrolytes.
Includes tools needed to launch high-throughput workflows 
for assessing properties of molecular using 
[QCArchive](http://docs.qcarchive.molssi.org/en/latest/),
create machine learning models using a [variety](https://schnetpack.readthedocs.io/en/stable/)
[of](https://github.com/NREL/nfp) 
[methods](http://www.qmlcode.org/),
and using the machine learning models to 
steer simulation campaigns with [Colmena](http://colmena.rtfd.org/).

Our goal is to create a software that is easy to deploy at different 
HPC centers and retool for new molecular applications.
The code itself is very experimental,
so expect the APIs to change frequently at this stage but
please do complain if documentation is lacking.

## Installation

The environment needed to run the tools is defined in `environment.yml`.
Create the environment with Miniconda by calling:

```shell
conda env create --file environment.yml --force
```

The environment should be run on a Linux system.
Detailed installation instructions will be provided
after merging this project with [Colmena](https://colmena.readthedocs.io/en/latest/installation.html).  
