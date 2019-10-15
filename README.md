# Electrolyte Design Workflows

Workflow tools for electrolyte design project: a collaboration between ExaLearn and JCESR.

## Installation

The environment needed to run the tools is defined in `environment.yml`.
Create the environment with Miniconda by calling:

```shell
conda env create --file environment.yml
```

## Design Principles

The Electrolyte Design Workflow (`edw`) library describes of the computations needed to design electrolytes from computation.

The calculations are broken into two categories:

  - "actions": Descriptions of specific computations (e.g., invoke NWCHem to relax a structure)
  - "workflows": Expressions of how to link the actions together to form larger computational workflows

### Actions

The "actions" are designed to describe simple steps in molecular design workflows
and is designed to be portable to different systems. 
Each action is defined by a Python function that takes common Python types in as input
and produces common types as an output.
The goal is to limit the amount of extra code needed to work with these functions, 
and to make them compatible with different workflow systems.

### Workflows

The "workflow" parts of the library define how to string the actions together using different workflow systems.
At present, we support:

- Parsl

#### Parsl

[Parsl](http://parsl-project.org/) is a Python library designed to simplify expressing and executing Python workflows.
The key component of Parsl workflows are "apps," which define specific tasks to be executed on external workflows.
The `edw.parsl.apps` module contains a list of these apps.

Parsl apps generate [futures](https://en.wikipedia.org/wiki/Futures_and_promises) that describe the outputs of calculations.
Futures as passed to other apps to form workflows.
The scripts that define a workflow are currently stored in home directory, for lack of a better home.
