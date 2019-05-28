#!/bin/sh

# Get number of threads from environment or set to default
if [ -z "$NTHREADS" ]; then
   NTHREADS=5
fi

echo "Using $NTHREADS threads"

# make sure OpenMP doesn't interfere with pytorch.multiprocessing
export OMP_NUM_THREADS=1

python3 lorentzian_embed_with_normalized_rank.py \
       -dim 10 \
       -lr 0.1 \
       -epochs 3000 \
       -negs 50 \
       -burnin 20 \
       -nproc "${NTHREADS}" \
       -distfn dist_lorentz \
       -dset eurovoc/eurovoc_closure.tsv \
       -fout eurovoc_10_0_01_with_no_normalized_rank.pth \
       -batchsize 50 \
       -eval_each 100 \
       -rin eurovoc/eurovoc_rank.txt \
       -beta 0.01 \
       -lambdaparameter 0.0 \
       -eta 150
