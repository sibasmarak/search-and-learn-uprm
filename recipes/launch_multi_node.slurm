#!/bin/bash
#SBATCH --account=a132
#SBATCH --partition=normal
#SBATCH --job-name=test_time_compute
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00

set -euo pipefail
set -x

export HF_HOME=/capstor/scratch/cscs/spanigra/huggingface
export RAY_IGNORE_UNHANDLED_ERRORS=1
# --- Recommended for HPC: avoid dashboard/metrics issues and placement deadlocks ---
export RAY_DISABLE_DASHBOARD=1
export RAY_DEDUP_LOGS=0

# Force Ray placement group to SPREAD across nodes (prevents "stuck at placement group")
export VLLM_DISTRIBUTED_EXECUTOR_CONFIG='{"placement_group_options":{"strategy":"SPREAD"}}'
export VLLM_WORKER_MULTIPROCESS_METHOD=spawn
export VLLM_RPC_TIMEOUT_MS=600000
export XDG_CACHE_HOME=/capstor/scratch/cscs/spanigra/.cache
export VLLM_CACHE_DIR=$XDG_CACHE_HOME/vllm

# export VLLM_USE_V1=0 (commented out)
# export VLLM_USE_RAY_COMPILED_DAG=0 (commented out)
# export VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE=ray (commented out)
# export VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM=0 (commented out)


HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
PORT=29500

# Get head IP from that node
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc "hostname -I | awk '{print \$1}'")
export HEAD_IP PORT
export RAY_ADDRESS="${HEAD_IP}:${PORT}"

echo "========================================="
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "HEAD_NODE: $HEAD_NODE"
echo "HEAD_IP: $HEAD_IP"
echo "RAY_ADDRESS: $RAY_ADDRESS"
echo "========================================="

run_node_task() {
  set -euo pipefail

  # --- IMPORTANT: conda init + activate inside the srun-launched shell ---
  source ~/.bashrc
  conda activate sal

  # (Optional) sanity checks
  which python
  which ray
  ray --version || true

  python -c "import vllm, torch; print(vllm.__version__, torch.__version__)"

  local my_ip
  my_ip=$(hostname -I | awk '{print $1}')

  echo "========================================="
  echo "ProcID: ${SLURM_PROCID} Host: $(hostname) IP: ${my_ip}"
  echo "RAY_ADDRESS: ${RAY_ADDRESS}"
  echo "========================================="

  trap 'ray stop || true' EXIT INT TERM

  # Make vLLM match Ray's IP choice on this node
  export VLLM_HOST_IP="$my_ip"

  if [[ "${SLURM_PROCID}" -eq 0 ]]; then
    echo "[HEAD] Starting Ray head on ${my_ip}:${PORT}"
    ray start --head \
      --node-ip-address="${my_ip}" \
      --port="${PORT}" \
      --num-cpus="${SLURM_CPUS_PER_TASK}" \
      --num-gpus=4 \
      --include-dashboard=false

    echo "[HEAD] Ray status:"
    ray status --address="${RAY_ADDRESS}" || true

    # Wait for workers
    for i in {1..60}; do
      n=$(ray status --address="${RAY_ADDRESS}" 2>/dev/null | grep -c "node_" || true)
      echo "[HEAD] Connected nodes: ${n} (attempt ${i}/60)"
      [[ "${n}" -ge "${SLURM_NNODES}" ]] && break
      sleep 1
    done

    echo "[HEAD] Final Ray status:"
    ray status --address="${RAY_ADDRESS}" || true

    # unset CUDA_VISIBLE_DEVICES
    echo "[HEAD] Running python on head..."
    python scripts/test_time_compute.py "$@"

    echo "[HEAD] Done; stopping Ray."
    ray stop || true
    exit 0
  else
    echo "[WORKER] Starting Ray worker on ${my_ip}, connecting to ${RAY_ADDRESS}"

    # Wait for head TCP port to be reachable.
    # If nc is not installed, replace with /dev/tcp check (see below).
    for i in {1..120}; do
      if command -v nc >/dev/null 2>&1; then
        nc -z "${HEAD_IP}" "${PORT}" >/dev/null 2>&1 && break
      else
        (echo >/dev/tcp/${HEAD_IP}/${PORT}) >/dev/null 2>&1 && break
      fi
      echo "[WORKER] Waiting for head ${HEAD_IP}:${PORT} (attempt ${i}/120)"
      sleep 1
    done

    ray start \
      --address="${RAY_ADDRESS}" \
      --node-ip-address="${my_ip}" \
      --num-cpus="${SLURM_CPUS_PER_TASK}" \
      --num-gpus=4 \
      --block
  fi
}

export -f run_node_task
export RAY_ADDRESS HF_HOME RAY_IGNORE_UNHANDLED_ERRORS

# Launch one task per node, pass args safely
srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NTASKS}" --ntasks-per-node=1 --export=ALL \
  bash -lc 'run_node_task "$@"' bash "$@"
