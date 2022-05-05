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
    entity: <your entity>
    models_project: <project with models to eval>
    eval_project: <project to store eval runs>
    start_date: 2022-04-28 # only evaluate runs after this date
    prefix_filters: ["mini-"] # only evaluate runs matching this prefix

outputDir: <desired output path>
evaluationFrequencySteps: 100000    # How often we evaluate checkpoints in terms of steps
checkFrequencySeconds: 600  # Checks the runs in wandb every ten minutes

downstreamTaskConfigs:
    MedParaSimp:
        path: /path/to/evaluation/folder # /home/user/mistral/downstream/textgen/gpt2
        environment: biomedical-nlg-eval # conda env for task evaluation
        hparams:   # hyper parameters for each executable of each task
            mini:  # tag for model type (e.g. all, mini, xs, small, medium, XL) ... put all for any model
                train_e2e.py:
                    learning_rate: 1e-4
                    bsz: 16
                gen_batch.py:
                    batch_size: 9
                    length: 400
    MedQA: 
        path: /path/to/evaluation/folder # /home/user/mistral/downstream/seqcls
        environment: biomedical-nlu-eval
        hparams:
            mini:
                run_multiple_choice.py:
                    learning_rate: [2e-5, 3e-5] # the cartesian product of all hparam settings will be evaluated
                    warmup_ratio: [0.25, 0.5]
```

Start the Auto Evaluator:

`python3 scripts/evaluation/auto_evaluator.py --config-path </path/to/YAML>` 


### GCP

The following are the instructions on how to deploy the auto-evaluation pipeline to GCP.

1. Create a VM: `n1-standard-8` machine with a V100.
1. SSH into the machine.
1. Install the Nvidia drivers by typing `y` and enter.
1. Clone the necessary GitHub repos:
    1. `git clone https://github.com/stanford-crfm/composer.git`
    1. `git clone https://github.com/stanford-crfm/mistral` (branch: `michi_pubmed_downstream`)
1. Create a screen session: `screen -S deploy`.    
1. Install the necessary dependencies:
    1. `cd composer`
    1. `git checkout autoeval`
    1. `python3 -m venv venv`
    1. `source venv/bin/activate`
    1. `pip install -r scripts/evaluation/requirements.txt`
    1. `pip install -e .`
1. Create the configuration YAML file (call it `config.yaml`). Here is an example config file:
   ```text
    wandb:
        apiKey: <Your API key>
        project: mosaic-gpt2
        entity: stanford-mercury
    
    outputDir: output
    evaluationFrequencySteps: 100000    # How often we evaluate checkpoints in terms of steps
    checkFrequencySeconds: 600  # Checks the runs in wandb every ten minutes
    
    downstreamTaskConfigs:
        pubMedQA: /home/tonyhlee/mistral/downstream/seqcls
        medQA: /home/tonyhlee/mistral/downstream/mc
        nerBC5CDR: /home/tonyhlee/mistral/downstream/tokcls
        nerEBMPICO: /home/tonyhlee/mistral/downstream/tokcls
    ```
1. Run `python3 scripts/evaluation/auto_evaluator.py --config-path config.yaml &> run.log`
1. Exit out of the screen session: `ctrl-ad`.
1. The auto evaluator should run forever. To check on the evaluator:
   1. Check `run.log` for logs: `tail -f run.log`.
   1. Check `output/state.json` to see what's been processed.
