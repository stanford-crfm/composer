# Auto Evaluator (WIP)

## Code Overview

Starts a process that runs infinitely observing runs that belongs to a wandb entity using the wandb api. 
Whenever a new  checkpoint is uploaded to wandb, the server downloads the checkpoint, untars it, 
convert it to a HF checkpoint, runs the evaluation and uploads the result to wandb. 

The process will update the state of each checkpoint and keep track of the ones the artifacts that already 
have been evaluated.

The downstream tasks are defined in `scrpits/evaluation/tasks`.

## Setup

Create a YAML file:

```text
wandb:
    apiKey: <your wandb API key>
    project: <your wandb project>
    entity: <your entity>

outputDir: <desired output path>
evaluationFrequencySteps: 100000    # How often we evaluate checkpoints in terms of steps
checkFrequencySeconds: 600  # Checks the runs in wandb every ten minutes

downstreamTaskPaths:
    pubMedQA: /path/to/evaluation/folder e.g. /mistral/downstream/seqcls
    medQA: /path/to/evaluation/folder e.g. /mistral/downstream/mc
    nerBC5CDR: /path/to/evaluation/folder e.g. /mistral/downstream/tokcls
    nerEBMPICO: /path/to/evaluation/folder e.g. /mistral/downstream/tokcls
```

Start the Auto Evaluator:

`python3 scripts/evaluation/auto_evaluator.py --config-path </path/to/YAML>` 


## GCP

TODO: Start a VM with a decent GPU.
