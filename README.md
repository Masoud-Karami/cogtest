# SSH Configuration Setup

# 1. Configure your SSH settings
```cat ~/.ssh/config
Host narval beluga graham cedar
    User <your_username>
    HostName %h.alliancecan.ca
    IdentityFile ~/.ssh/your_private_key
```

# 2. Connect to a cluster
```ssh narval```

# 3. Transfer files to remote cluster
```scp local_file narval:work/```

# 4. Add your public SSH key to each cluster separately

# Compute Canada Setup

# Log into a Compute Canada cluster
```ssh username@beluga.computecanada.ca```

# Create a directory for HuggingFace models
```mkdir -p ~/scratch/huggingface/meta-llama```

# Load and install Git LFS
```module load git-lfs```
```git lfs install```

# Clone LLaMA repository (requires access token)
```git lfs clone https://huggingface.co/meta-llama/Llama-2-7b-hf```

# Set HuggingFace cache directory
```unset TRANSFORMERS_CACHE```
```export HF_HOME=~/scratch/huggingface/```

# Git Configuration

# Set remote URL and check status
```cd ~/scratch/llms_serialmemory```
```git remote set-url origin git@github.com:Masoud-Karami/cogtest.git```
```git remote -v```

# Work on correct branch
```git checkout newtask-serialmemory```
```git pull --rebase origin newtask-serialmemory```
```git push origin newtask-serialmemory```

# Clone CogBench baseline (if needed)
```cd ~/scratch```
```git clone https://github.com/mamerzouk/CogBench```
```cd CogBench```
```git checkout computecanada-no-third-party```

# Experimental Environment Setup

# Allocate a compute node (temporary)
```srun --pty --cpus-per-task=8 --mem=16G --gres=gpu:1 --time=04:00:00 bash```

# Inside the node, set up Python environment
```cd ~/scratch/CogBench```
```module load python/3.10.13```
```module load python/3.11.5```
``` module load python/3.12.4```
```module load python/3.13.2```

```
python -m venv ~/envs/temp_env
source ~/envs/hf-llama/bin/activate
pip install --upgrade pip
pip install transformers accelerate huggingface_hub   
```

```source llamaenv/env/bin/activate```

# Install Python packages
```pip install --no-index --upgrade pip```
```pip install --no-index transformers accelerate huggingface_hub    #for downloading hf meta llama models``` 
```pip install --no-index -r requirements.txt```
```git config --global credential.helper store```
```huggingface-cli login```
```git lfs install```

# Export Python path for local imports
```export PYTHONPATH=$(pwd)```

# Run Serial Memory Experiments
```python3 full_run.py --engine llama-2-7```
