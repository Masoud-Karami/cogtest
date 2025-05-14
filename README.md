# Replicating CogBench: a large language model walks into a psychology lab

<!---!
[Overview Figure](./overview_figure.png)
Cheat Sheet: Adding Math Notation to Markdown (https://www.upyesp.org/posts/makrdown-vscode-math-notation/)
Markdown Cheat Sheet Basic (https://www.markdownguide.org/cheat-sheet/)

-->

# SSH configuration file

## [avoid typing command each time](https://docs.alliancecan.ca/wiki/SSH_configuration_file) and [transfere file](https://docs.alliancecan.ca/wiki/Transferring_data)

ssh -i ~/.ssh/your_private_key username@narval.alliancecan.ca

1. add the following to `~/.ssh/config` on your local machine

- ```Host narval beluga graham cedar
   User username
   HostName %h.alliancecan.ca
   IdentityFile ~/.ssh/your_private_key```

- `[name@yourLaptop ~] ssh narval`

2. Then Transferring data would be easier

- `[name@yourLaptop ~] scp local_file narval:work/`

3. you need to install your `public SSH key` on each cluster separately


```bash
python3 full_run.py --engine claude-1 --only_analysis
```

```bash
python3 full_run.py --engine random --compare_with gpt-4 claude-1
```

## Contributing
`.csv`

## Reference

[ ](https:/).

# wisellm: a large language model walks into a psychology lab

This repository contains the code for wisellm, a cognitive psychology benchmark. The project is structured into three main folders: `Experiments`, `llm_utils`, and `Analysis`.

## Experimentsexport HF_HOME=/blabla/cache/

The `Experiments` folder contains different experiments that you can run. Each subfolder corresponds to a different cognitive psychology experiment. The folder contains a README.md file with instructions on how to run experiments and compute the behavioral metrics.

## LLM Utils

The `llm_utils` folder contains scripts for different LLMs. If you want to add your own LLM, you should create a new script in this folder. Please refer to the `llm_utils` folder README.md for more details on how to do this.

## Analysis

The `Analysis` folder contains scripts that merge information from the LLMs and the Experiments scores. You are encouraged to add your own analysis scripts to this folder.

## Requirements

Before running any scripts, please install all the dependencies listed in the `requirements.txt` file.

## Running the Entire Benchmark

To run the entire benchmark for a chosen LLM, you can use the `full_run.py` script. This script will run all the experiments, store the required values, and print and plot the main metrics.

Before running the script, make sure that your LLM is recognized in `llm_utils`. If it's not, you'll need to add it there first.

Please note that the fitting of scores is generally fast for all experiments, except for the InstrumentalLearning experiment which can be very slow. Please be patient when running this experiment's fitting.

Here's how you can use the script with the `random` agent as an example:

```bash
python3 full_run.py --engine random
```(error)
You can use the `--only_analysis` flag if you only want to run the analysis and skip the experiment running and storing steps. This can be useful if you have already run the experiments and just want to see the analysis results or if you want to just run the analysis for the LLMs that have already been run (for which the data is already stored). Here is how you can use the script with the agent (here claude-1 as example) and the --only_analysis flag:
```bash
python3 full_run.py --engine claude-1 --only_analysis
```

After the analysis, a summary table is printed with the scores (before normalization) for the chosen agent, as well as human and random agents and reference scores for the models specified with the `--compare_with` flag (default: gpt-4, claude-2). The performance and behavior normalized scores versus the models specified with the `--compare_with` flag are also plotted. The plots are saved in the `./Analysis/plots/phenotypes/full_runs{interest}.pdf` directory.

You can specify the models to compare against when running the script. For example, to compare against gpt-4 and claude-1, you would use the `--compare_with` flag like this:

```bash
python3 full_run.py --engine random --compare_with gpt-4 claude-1 
```

## computecanada
  ### Login to a Compute Canada server
  ```ssh username@beluga.computecanada.ca```

  ### Create a directory for HuggingFace models
  ```mkdir ~scratch/huggingface/meta-llama```
  
  ```cd ~scratch/huggingface/meta-llama```

  ### Load and install git LFS
  ```module load git-lfs```
  
  ```git lfs install```

  ### Load proper branch
  ```git branch -a```
  
  ```git checkout <newtask-serialmemory>```

  ### Download Llama (you will need your username and token)
  ```git lfs clone https://huggingface.co/meta-llama/Llama-2-7b-hf```

  ### Set the HuggingFace environment variable
  ~~```export TRANSFORMERS_CACHE=~/scratch/huggingface/```~~ [error](https://stackoverflow.com/questions/63312859/how-to-change-huggingface-transformers-default-cache-directory)

  ```os.environ['HF_HOME'] = '~/scratch/huggingface/'```
  
  If TRANSFORMERS_CACHE still exists (`echo $TRANSFORMERS_CACHE` or 
`echo $HF_HOME`), unset it:
  `unset TRANSFORMERS_CACHE`
  ```export HF_HOME=~/scratch/huggingface/```

  ### Clone my repository (check the commits)
  ``c``d ~/scratch```
  ```git clone https://github.com/mamerzouk/CogBench```
  
  ```git checkout computecanada-no-third-party```

  ### Move to a compute note (this allocation is temporary and just for tests)
  ```srun --pty --cpus-per-task=8 --mem=16G --gres=gpu:1 --time=03:00:00 bash```

  ### When you're in the compute node, load python, virtualenv and install the necessary libaries
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