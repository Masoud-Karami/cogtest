# SSH Configuration Setup

~/.ssh/config
```
Host narval beluga graham cedar
    User <your_username>
    HostName %h.alliancecan.ca
    IdentityFile ~/.ssh/your_private_key~/.ssh/config

ssh narval
module load git-lfs
git lfs install
git lfs clone https://huggingface.co/meta-llama/Llama-2-7b-hf
unset TRANSFORMERS_CACHE```
export HF_HOME=~/scratch/huggingface/```
srun --pty --cpus-per-task=8 --mem=16G --gres=gpu:1 --time=04:00:00 bash
cd ~/scratch/CogBench
module load python/3.10.13
module load python/3.11.5
module load python/3.12.4
module load python/3.13.2
python -m venv ~/envs/cccbenvs
source ~/envs/hf-llama/bin/activate

du -sh *

pip install --no-index --upgrade pip
pip install --no-index transformers accelerate huggingface_hub    #for downloading hf meta llama models
git config --global credential.helper store
huggingface-cli login
git lfs install

export PYTHONPATH=$(pwd)
python Experiments/SerialMemoryTask/query.py --list_lengths 10 --starting_conditions constant --max_trials 4 --num_sessions 1
```
