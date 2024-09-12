#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080 # partition (queue)
#SBATCH -c 8 # number of cores
#SBATCH -o /work/dlclarge1/engell-crypto_ml/log/%x.%N.%j.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e /work/dlclarge1/engell-crypto_ml/log/%x.%N.%j.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J test_transformer # sets the job name. If not specified, the file name will be used as job name

# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Job to perform
python ~/Crypto_ML/Transformer/scripts/test_model.py /work/dlclarge1/engell-crypto_ml/ transformer

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";