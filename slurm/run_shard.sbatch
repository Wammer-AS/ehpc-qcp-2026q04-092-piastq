#!/bin/bash
#SBATCH --job-name=piastq_bench
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --time=01:00:00
# Pass --exclude=<gpu_node> on the sbatch command line if the partition includes GPU nodes.

set -euo pipefail
unset PYTHONPATH
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

REPO_DIR="${REPO_DIR:-$PWD}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"

source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR"

OUT_DIR="${OUT_DIR:-$REPO_DIR/artifacts_20261127}"
mkdir -p "$OUT_DIR"

SHARD="${1:?Usage: sbatch slurm/run_shard.sbatch <shard_id>}"
echo "Starting shard=$SHARD job=${SLURM_JOB_ID:-local} at $(date -Is)"

case "$SHARD" in
  node1)
    if [ ! -f "$OUT_DIR/point2_gate_mapping.json" ]; then
      OMP_NUM_THREADS=32 python scripts/run_piastq_benchmark.py --mode gate_map --output-dir "$OUT_DIR" --workers 1 --aer-threads 32
    fi
    OMP_NUM_THREADS=64 python scripts/run_piastq_benchmark.py --mode workflow_demo --output-dir "$OUT_DIR" --workers 1 --aer-threads 64
    OMP_NUM_THREADS=2 python scripts/run_piastq_benchmark.py --mode dev_n8_12 --output-dir "$OUT_DIR" --workers 48 --aer-threads 2
    OMP_NUM_THREADS=5 python scripts/run_piastq_benchmark.py --mode dev_n16 --output-dir "$OUT_DIR" --workers 24 --aer-threads 5
    OMP_NUM_THREADS=5 python scripts/run_piastq_benchmark.py --mode shot_scaling_n8_12_16 --output-dir "$OUT_DIR" --workers 24 --aer-threads 5
    OMP_NUM_THREADS=8 python scripts/run_piastq_benchmark.py --mode dev_3body --output-dir "$OUT_DIR" --workers 16 --aer-threads 8
    ;;
  node2)
    OMP_NUM_THREADS=5 python scripts/run_piastq_benchmark.py --mode dev_n20_part1 --output-dir "$OUT_DIR" --workers 24 --aer-threads 5
    ;;
  node3)
    OMP_NUM_THREADS=5 python scripts/run_piastq_benchmark.py --mode dev_n20_part2 --output-dir "$OUT_DIR" --workers 24 --aer-threads 5
    ;;
  node4)
    OMP_NUM_THREADS=8 python scripts/run_piastq_benchmark.py --mode shot_scaling_n20 --output-dir "$OUT_DIR" --workers 16 --aer-threads 8
    ;;
  *)
    echo "Unknown shard: $SHARD" >&2
    exit 2
    ;;
esac

echo "Finished shard=$SHARD job=${SLURM_JOB_ID:-local} at $(date -Is)"
