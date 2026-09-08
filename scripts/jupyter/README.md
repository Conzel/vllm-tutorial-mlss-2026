# Running the tutorial notebooks on MPI

The notebooks need a GPU. These scripts start a Jupyter Lab server on an MPI
compute node (HTCondor) and tunnel it to your laptop, so you edit locally in the
browser while the kernel runs on an A100/H100.

## Usage

    ./scripts/jupyter/mpi-jupyter --sync   # push local changes, then start
    ./scripts/jupyter/mpi-jupyter          # start or reattach
    ./scripts/jupyter/mpi-jupyter --stop   # end the session

Prints `http://localhost:8888/lab?token=...`. Ctrl-C closes the tunnel but
leaves the job running -- re-run to reattach.

## Before the workshop: cache the models

Compute nodes run with `HF_HUB_OFFLINE=1` and have no internet, so every model a
notebook loads must already be in `$HF_HOME`. Fetch them once from the login
node, which does have internet:

    ./scripts/prefetch-models           # download (idempotent -- skips what is cached)
    ./scripts/prefetch-models --check   # just report what is present

Notebook 02 adds `Qwen2.5-1.5B-Instruct` (the verifier) alongside the 0.5B model
used elsewhere. Notebook 02's first cell re-checks the cache and stops with the
command above if anything is missing.

## How it works

1. `jupyter.sub` requests one A100/H100 (bid 25, 4 cpus, 32GB).
2. `jupyter_job.sh` runs on the node: picks a free port and a random token,
   writes them to `.jupyter/session-<job-id>.env` on `/lustre`, starts Jupyter.
3. `mpi-jupyter` reads that file over SSH and opens
   `ssh -L 8888:<node>:<port> mpi-rsync`.

Singularity shares the host network namespace, so a port opened in the container
is a port on the compute node, and the login node can reach compute nodes.

## Configuration

    MPI_HOST        ssh host for the login node  (default: mpi-rsync)
    MPI_REPO_DIR    checkout path on /lustre
    MPI_LOCAL_PORT  local port                   (default: 8888)
    MPI_BID         condor bid                   (default: 25, A100 minimum)

`mpi-rsync` is used rather than `mpi`/`m` because those carry a `RemoteCommand`
that starts an interactive nushell, which breaks `ssh host "cmd"`.

## First-time setup on the cluster

Compute nodes have **no internet access**, so the environment and the model
weights must exist before the job starts. On the login node:

    rsync -av ~/vllm-tutorial/ mpi-rsync:/lustre/home/aconzelmann/vllm-tutorial/

The venv and the model cache must live on `/fast`, **not** in your home
directory -- the home quota is ~150GB and already full, while `/lustre/fast`
has TBs free. torch alone is 2.5GB installed.

    ssh mpi
    mkdir -p /fast/aconzelmann/vllm-tutorial/{wheels,pip-cache,tmp}
    cd /fast/aconzelmann/vllm-tutorial
    PIP_CACHE_DIR=$PWD/pip-cache TMPDIR=$PWD/tmp \
      python3 -m venv venv && venv/bin/pip install vllm jupyterlab

    HF_HOME=/fast/aconzelmann/hf_home hf download Qwen/Qwen2.5-0.5B-Instruct

If a large wheel (torch is 527MB) dies mid-download with `incomplete-download`,
pip's resume does not recover. Fetch it with `curl -C -` into `wheels/` and
`pip install` the local file, then re-run the pip command.

To use a Singularity image instead of the venv, set `SIF=/path/to/image.sif` in
the `environment =` line of `jupyter.sub`; `jupyter_job.sh` switches
automatically.

## vLLM in a kernel

`jupyter_job.sh` exports three variables that vLLM needs on these nodes. They are
not optional -- drop any one and `LLM(...)` fails in the notebook. Verified by
starting a kernel through the server's API and generating text on an A100.

| Variable | Without it |
|---|---|
| `VLLM_ENABLE_V1_MULTIPROCESSING=0` | The notebook's benchmark cell initializes CUDA, so vLLM switches to `spawn`; spawn cannot re-import `__main__` in a kernel and raises "An attempt has been made to start a new process before the current process has finished its bootstrapping phase". In-process mode uses no subprocess. |
| `CUDA_HOME=$VENV/.../nvidia/cu13` | "Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist" -- compute nodes have no system CUDA toolkit, but pip installed nvcc into the venv. |
| `VLLM_USE_FLASHINFER_SAMPLER=0` | FlashInfer JIT-compiles sampling kernels on first use and its vendored CCCL headers reject that nvcc: "CUDA compiler and CUDA toolkit headers are incompatible". It also compiles into `~/.cache`, which is over quota. vLLM's native sampler needs no compiler. |

If you run vLLM outside this Jupyter setup (a plain batch job, say), you need
the same three.

Two consequences of in-process mode worth knowing when teaching:

- The engine holds GPU memory for the kernel's lifetime. Creating a second
  `LLM` without `del llm; torch.cuda.empty_cache()` will OOM. Consider
  `gpu_memory_utilization=0.5` rather than `0.8` in the notebook.
- With `enforce_eager=False`, CUDA-graph capture happens in-process, so the
  first `generate()` blocks the kernel noticeably longer.

## Notes

- Jupyter binds `0.0.0.0` so the login node can reach it; the random token is
  the only access control. If policy forbids that, set `--ip=127.0.0.1` in
  `jupyter_job.sh` and tunnel through `condor_ssh_to_job` instead.
- vLLM's kernel caches go to `/tmp` on the node (local, fast) rather than
  `/lustre`, which is slow for many small files.
- `.jupyter/` holds live server tokens -- keep it out of version control.
- `condor_q` on this cluster lists the **whole pool**, not just your jobs, so
  `mpi-jupyter` filters with `condor_q $(whoami)` plus a `JobBatchName`
  constraint. Don't drop either half.
- Condor expands `$(ClusterId)` in an `environment =` line to the GlobalJobId
  form (`sched#17522601.0#7734`), which does not match what `condor_q` reports.
  `jupyter_job.sh` therefore reads the real `ClusterId.ProcId` from
  `$_CONDOR_JOB_AD`. Keep that if you edit the script.
- `initialdir` is set to the repo root in `jupyter.sub`, so every relative path
  (the `.jupyter/` logs, the executable) resolves against `REPO_DIR` rather than
  the directory you submitted from.
