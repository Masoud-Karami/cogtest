# Replicating CogBench: a large language model walks into a psychology lab

<!---!
[Overview Figure](./overview_figure.png)
Cheat Sheet: Adding Math Notation to Markdown (https://www.upyesp.org/posts/makrdown-vscode-math-notation/)
Markdown Cheat Sheet Basic (https://www.markdownguide.org/cheat-sheet/)

-->

# SSH configuration file

## [avoid typing command each time](https://docs.alliancecan.ca/wiki/SSH_configuration_file)

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

## Cite
