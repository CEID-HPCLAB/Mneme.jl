#!/bin/bash

NUM_MPI_PROCESSES=8
JULIA_WORKER_THREADS=2

export PATH="$HOME/.julia/bin:$PATH"
export PATH="$HOME/.julia/packages/MPI/bin:$PATH"

echo "Running with $NUM_MPI_PROCESSES MPI processes and $JULIA_WORKER_THREADS Julia threads..."

mpiexecjl -np "$NUM_MPI_PROCESSES" julia -t "$((JULIA_WORKER_THREADS + 1)),0" \
    --project="." example.jl