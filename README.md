# SSH Configuration Setup

~/.ssh/config
```
Host narval beluga graham cedar
    User <your_username>
    HostName %h.alliancecan.ca
    IdentityFile ~/.ssh/your_private_key~/.ssh/config

ssh narval

# to download a model:
huggingface-cli login
git lfs install
module load git-lfs
git lfs clone https://huggingface.co/meta-llama/Llama-2-7b-hf


unset TRANSFORMERS_CACHE
export HF_HOME=~/scratch/huggingface/
srun --pty --cpus-per-task=8 --mem=16G --gres=gpu:1 --time=04:00:00 bash
module load python/3.10
module load python/3.11
module load python/3.12
cd ~/scratch/llms_serialmemory

virtualenv --no-download $SLURM_TMPDIR/cccbvenv # or ~/envs/cccbenvs
source $SLURM_TMPDIR/cccbvenv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
pip freeze --local > requirements.txt


pip install --no-index transformers accelerate huggingface_hub    #for downloading hf meta llama models
git config --global credential.helper store


export PYTHONPATH=$(pwd)
python Experiments/SerialMemoryTask/query.py --list_lengths 10 --starting_conditions constant --max_trials 4 --num_sessions 1
PYTHONPATH=$(pwd) python Experiments/SerialMemoryTask/query.py --engine TinyLlama-1.1B-Chat-v1.0 --add_distr --debug
```
