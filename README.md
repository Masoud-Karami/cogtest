# SSH Configuration Setup

~/.ssh/config
```
Host narval beluga graham cedar
    User <your_username>
    HostName %h.alliancecan.ca
    IdentityFile ~/.ssh/your_private_key~/.ssh/config
ssh narval
scp local_file narval:work/
ssh username@beluga.computecanada.ca
mkdir -p ~/scratch/huggingface/meta-llama
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
python -m venv ~/envs/temp_env
source ~/envs/hf-llama/bin/activate
pip install --upgrade pip
pip install transformers accelerate huggingface_hub   
source llamaenv/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index transformers accelerate huggingface_hub    #for downloading hf meta llama models
pip install --no-index -r requirements.txt
git config --global credential.helper store
huggingface-cli login
git lfs install
export PYTHONPATH=$(pwd)
python3 full_run.py --engine llama-2-7
```
