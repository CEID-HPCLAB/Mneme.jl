#!/bin/bash

NUM_MPI_PROCESSES=4
JULIA_WORKER_THREADS=2

export PATH="$HOME/.julia/bin:$PATH"
export PATH="$HOME/.julia/packages/MPI/bin:$PATH"

if ! command -v julia >/dev/null 2>&1; then
    echo "ERROR: julia not found in PATH." >&2
    exit 1
fi

JULIA_VERSION=$(julia --startup-file=no -e 'print(VERSION)')
echo "Julia: $(command -v julia) (version $JULIA_VERSION)"

NTHREADS=$((JULIA_WORKER_THREADS + 1))

if julia --startup-file=no -e 'exit(VERSION >= v"1.12" ? 0 : 1)'; then
    THREAD_FLAG="$NTHREADS,0"
else
    THREAD_FLAG="$NTHREADS"
fi

echo "Running with $NUM_MPI_PROCESSES MPI processes and $NTHREADS Julia threads..."

mpiexecjl -np "$NUM_MPI_PROCESSES" julia -t "$THREAD_FLAG" --project="." example.jl