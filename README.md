# Hypercube Allreduce

## Usage

Just make your target (`make debug` or `make release`). The name of the executable is
`allreduce`. Then execute the mpi benchmark depending on your environment. E.g., use
`mpirun -np $numofprocesses allreduce`.

## Implementation

The code technically requires a C++11 compiler. Make uses the compiler wrapper mpic++.
You can change the compiler in the Makefile though.
