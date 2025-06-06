
<!---!

-->

# SSH configuration file

## [avoid typing command each time](https://docs.alliancecan.ca/wiki/SSH_configuration_file) and [transfere file](https://docs.alliancecan.ca/wiki/Transferring_data)

ssh -i ~/.ssh/your_private_key username@narval.alliancecan.ca

1. Add the following to `~/.ssh/config` on your local machine

- ```console
        Host narval beluga graham cedar
            User <username>
            HostName %h.alliancecan.ca
            IdentityFile ~/.ssh/your_private_key```

- `[name@yourLaptop ~] ssh narval`

2. Then transferring data would be easier

- `[name@yourLaptop ~] scp local_file narval:work/`

3. You need to install your `public SSH key` on each cluster separately

## computecanada
  ### Login to a Compute Canada server
  ```ssh username@beluga.computecanada.ca```

  ### Create a directory for HuggingFace models
  ```mkdir ~scratch/huggingface/meta-llama```

  ### Load and install git LFS
  ```module load git-lfs```
  
  ```git lfs install```

  ### Load proper branch
  ```git branch -a```
  
  ```git checkout newtask-serialmemory```

  ### Download Llama (you will need your username and token)
  ```git lfs clone https://huggingface.co/meta-llama/Llama-2-7b-hf```

  ### Set the HuggingFace environment variable
  ~~```export TRANSFORMERS_CACHE=~/scratch/huggingface/```~~ [error](https://stackoverflow.com/questions/63312859/how-to-change-huggingface-transformers-default-cache-directory)

  ```os.environ['HF_HOME'] = '~/scratch/huggingface/'```
  
  If TRANSFORMERS_CACHE still exists (`echo $TRANSFORMERS_CACHE` or 
`echo $HF_HOME`), unset it:
  `unset TRANSFORMERS_CACHE`
  ```export HF_HOME=~/scratch/huggingface/```git remote set-url origin git@github.com:Masoud-Karami/cogtest.git


  when in Home/cogtest, ```export PYTHONPATH=$(pwd)```
## GIT

```bash 
git remote set-url origin git@github.com:Masoud-Karami/cogtest.git 
git remote -v 
git checkout newtask-serialmemory 
git pull --rebase origin newtask-serialmemory 
git push origin newtask-serialmemory 
```


  ### Clone my repository (check the commits)
  ``c``d ~/scratch```
  ```git clone https://github.com/mamerzouk/CogBench```
  
  ```git checkout computecanada-no-third-party```

  ### Move to a compute note (this allocation is temporary and just for tests)
  ```srun --pty --cpus-per-task=8 --mem=16G --gres=gpu:1 --time=03:00:00 bash```

  ### When you're in the compute node, load Python, virtualenv, and install the necessary libraries
  ```cd ~/scratch/CogBench```
  
  ```module load python/3.11.5```
  
  ```virtualenv --no-download $SLURM_TMPDIR/env```
  
  ```source $SLURM_TMPDIR/env/bin/activate```

  ```pip install --no-index --upgrade pip```
  
  ```pip install --no-index accelerate```
  
  ```pip install --no-index -r requirements.txt```

  ### Run the experiments with llama2-7b
  ```python3 full_run.py --engine llama-2-7```

the code is in a way to satisfy the following example strictly.
Consider the noisy list introduced one line as a separate prompt to the model. 
001. "#job"
002. "@$in"
003. "[DISTRACTOR] Distracto%%"
004. "[DISTRACTOR] Xyzon@$@"
005. "[DISTRACTOR] ##&Xyzon*"
006. "tailor"
007. "[DISTRACTOR] Distracto&*"
008. "*#'s**"
009. "[DISTRACTOR] ^*$Obscure~$$"
010. "[DISTRACTOR] ##Fuzz"
In the test model is free to recall the word by its knowledge. We shouldn't change or modify its responses. So for example consider the following responses
001. "#job"                                                               #job
002. "@$in"                                                              in
003. "[DISTRACTOR] Distracto%%"                       Distracto                                      
004. "[DISTRACTOR] Xyzon@$@"                          [the model is silent here as output indicated by <<SILENT>>]                                    
005. "[DISTRACTOR] ##&Xyzon*"                        [the model is silent here as output indicated by <<SILENT>>]                                       
006. "tailor"                                                             tailor 
007. "[DISTRACTOR] Distracto&*"                       Distracto                                       
008. "*#'s**"                                                            #'s  
009. "[DISTRACTOR] ^*$Obscure~$$"               [the model is silent here as output indicated by <<SILENT>>]                                                
010. "[DISTRACTOR] ##Fuzz"                              [the model is silent here as output indicated by <<SILENT>>] 

Then the code matches the exact original output with the clean input. 
- In case the output  <<SILENT>> is matched with [DISTRACTOR] labeled, TRUE RECALLED, 
- in case the original output (e.g #'s)  doesn't match the clean targetted word, FALSE RECALL