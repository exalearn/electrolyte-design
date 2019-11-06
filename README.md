# Electrolyte Design Workflows

Workflow tools for electrolyte design project: a collaboration between ExaLearn and JCESR.
The tools are designed to scalable on leadership-class computing systems
and to maximize use of existing quantum chemistry workflow libraries.

**Context**: This library is intended as a prototype to identify needs for the electrolyte design 
problem and not to present a cohesive solution.
Our current goal is to offload/combine elements of this library with other tools. 
A current list of projects that EDW should sync with include:
 - [QCArchive](https://qcarchive.molssi.org/)
 - [qmwf](https://github.com/alvarovm/qmwf)
 - [QTC](https://github.com/PACChem/QTC)
 - [Balsam](https://balsam.readthedocs.io/en/latest/tutorial/nwchem.html)
 - [Atomate](https://github.com/hackingmaterials/atomate)
 
(If you're reading this and know of something else, please issue a PR to add it)

## Installation

The environment needed to run the tools is defined in `environment.yml`.
Create the environment with Miniconda by calling:

```shell
conda env create --file environment.yml --force
```

The environment needs to run on a Linux system.
There are issues around using Parsl on Mac OS X and Windows, 
and issues with MongoDB on Windows Subsystem for Linux. 

## Using EDW

There are two key components of EDW: a database holding the currently-known molecular properties
and run-scripts that perform new calculations and update the database. 

### Database

The database is created using MongoDB and intended to be portable.
The database is currently running on local hardware at Argonne and accessible by 
creating a SSH tunnel to that resource.

The database has two collections: a GridFS collection with all calculation files
and a collection with BSON records with one record per molecule.

### Run Scripts

The "run scripts" a (presently ad hoc) collection of Python scripts aht 
each perform a specific type of calculation needed for electrolyte design.
They all roughly follow the procedure:

1. Connect to MongoDB and retrieve molecules that are ready for this calculation
2. Configuring Parsl to run on a certain resource
3. Assembling and submitting Calculations to Parsl to manage
4. As calculations complete, writing output files or updated records to the database

The scripts are designed to run on the login node for the cluster and 
use Parsl for interacting with the job scheduler.

## Design Principles

The Electrolyte Design Workflow (`edw`) library provides the tools
needed to design electrolytes from atomic-scale simulations and machine learning.
As noted in the instructions for using ``edw``, EDW is composed of two software 
components: a database and components of a workflow engine.

### Workflow Engine
The workflow tools are broken into two categories:

  - "actions": Descriptions of specific computations (e.g., write an NWCHem input file)
  - "workflows": Expressions of how to link the actions together to form larger computational workflows

### Actions

The "actions" are designed to describe simple steps in molecular design workflows
and is designed to be portable to different systems. 
Each action is defined by a Python function that takes common Python types in as input
and produces common types as an output.
The goal is to limit the amount of extra code needed to work with these functions, 
and to make them compatible with different workflow systems.

There are a specific class of actions that store results into databases.
These actions might be moved into their own module.

### Workflows

The "workflow" parts of the library define how to string the actions together using different workflow systems.

At present, we only support Parsl.
[Parsl](http://parsl-project.org/) is a Python library designed to simplify expressing and executing Python workflows.
The key component of Parsl workflows are "apps," which define specific tasks to be executed on external workflows.
The `edw.parsl.apps` module contains a list of these apps.
Parsl apps generate [futures](https://en.wikipedia.org/wiki/Futures_and_promises) that describe the outputs of calculations.
Futures as passed to other apps to form workflows.

The ``edw`` library contains a series of Parsl apps that perform simple tasks that should be performed
in parallel (e.g., running NWChem) and the run-scripts stitch them together into appropriate workflows.
