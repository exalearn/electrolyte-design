# Evaluating NWChem Ensembles

Running large numbers of NWChem computations has caused problems with scaling our application.
The architecture of ALCF's Theta demands the NWChem task to be launched using an "aprun" call 
from an application on the MOM node, which we accomplish by placing a Parsl executor on the MOM node.
The MOM node has limited reosurces, which we have overloaded when launching too many NWChem tasks.
Here, we explore how to reduce the computational requirements of applications on the MOM Node.

## The Test

We have a simple application which runs an ensemble of many NWChem runs.
The application creates a certain number of NWChem workers and continually 
feeds new tasks to them as they complete previous jobs.
You are able to change how many workers are launched and the nature of these workers
in the application (e.g., whether a worker is a separate process or thread).

The application records the overheads for each NWChem run and the resource usage 
on the MOM node.
