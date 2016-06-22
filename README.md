# Hypercube Allreduce

## Usage

Just make your target (`make debug` or `make release`). The name of the executable is
`allreduce`. Then execute the mpi benchmark depending on your environment. E.g., use
`mpirun -np $numofprocesses allreduce`.
Just include `ssssort.h` and use `ssssort::ssssort(Iterator begin, Iterator end)`.
Or, if you want the output to be written somewhere else, use the version with
three iterators: `ssssort::ssssort(InputIt begin, InputIt end, OutputIt out_begin)`.
Note that the input range will be in an arbitrary order after calling this.

## Implementation

The code technically requires a C++11 compiler. Make uses the compiler wrapper mpic++.
You can change the compiler in the Makefile though.
