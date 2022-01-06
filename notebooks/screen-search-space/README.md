# Screen Search Space

We plan to use large resources of molecules as the basis for a search and can elimiate many of the molecules from the start.
Some may have too large of a molecular weight to be economically-feasible and others may possess problematic substructures, 
such as acid groups that will lead to reacting with the solvents.
This folder contains scripts that perform quick screenings.

Running the screens on large search spaces is easy with chemoinformatics toolkits, like RDKit, but running them in parallel has a few challenges.
The largest is that storing the entire search space in memory is problematic.
So, we implement an out-of-core strategy where we gradually read molecules from the search space and send them out to workers as tasks complete.
That way, we do not have the whole dataset in memory at any one time.

## Running the Program

The `screen.py` program implements the parallel screening algorithm with [Colmena](https://colmena.rtfd.org). 
It requires a path to the search space of molecules and a description of what functional groups to screen out as a YAML file:

```yaml
allowed_elements: [C, H, O, N, F, S]  # List of allowed elements
bad_smarts:  # SMARTS strings that match substructures we want to avoid
  # Acids
  - "[CX3](=O)[OX1H0-,OX2H1]"
  - "[CX3](=O)[OX2H1]"
```

Call `python screen.py --help` for the full options. 
