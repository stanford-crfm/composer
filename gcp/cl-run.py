# launches a codalab instance with a docker image and runs a run yaml a la mcli run
import argparse
import os

import hiyapyco
import uuid
import tempfile
import subprocess
import shlex

# cl-run.py -f my-yaml
# commands that CL run understands:
"""
  -h, --help            show this help message and exit
  -w WORKSHEET_SPEC, --worksheet-spec WORKSHEET_SPEC
                        Operate on this worksheet ([(<alias>|<address>)::](<uuid>|<name>)).
  -a AFTER_SORT_KEY, --after_sort_key AFTER_SORT_KEY
                        Insert after this sort_key
  -m, --memoize         If a bundle with the same command and dependencies already exists, return it instead of creating a new one.
  -i, --interactive     Beta feature - Start an interactive session to construct your run command.
  -n N, --name N        Short variable name (not necessarily unique); must conform to ^[a-zA-Z_][a-zA-Z0-9_\.\-]*$.
  -d D, --description D
                        Full description of the bundle.
  --tags [TAG ...]      Space-separated list of tags used for search (e.g., machine-learning).
  --allow-failed-dependencies
                        Whether to allow this bundle to have failed or killed dependencies.
  --request-docker-image REQUEST_DOCKER_IMAGE
                        Which docker image (either tag or digest, e.g., codalab/default-cpu:latest) we wish to use.
  --request-time REQUEST_TIME
                        Amount of time (e.g., 3, 3m, 3h, 3d) allowed for this run. Defaults to user time quota left.
  --request-memory REQUEST_MEMORY
                        Amount of memory (e.g., 3, 3k, 3m, 3g, 3t) allowed for this run.
  --request-disk REQUEST_DISK
                        Amount of disk space (e.g., 3, 3k, 3m, 3g, 3t) allowed for this run. Defaults to user disk quota left.
  --request-cpus REQUEST_CPUS
                        Number of CPUs allowed for this run.
  --request-gpus REQUEST_GPUS
                        Number of GPUs allowed for this run.
  --request-queue REQUEST_QUEUE
                        Submit run to this job queue.
  --request-priority REQUEST_PRIORITY
                        Job priority (higher is more important). Negative priority bundles are queued behind bundles with no specified priority.
  --request-network     Whether to allow network access.
  --exclude-patterns [EXCLUDE_PATTERNS ...]
                        Exclude these file patterns from being saved into the bundle contents.
  --store STORE         The name of the bundle store where bundle results should be initially uploaded. If unspecified, an optimal available bundle store will be chosen.
  -e, --edit            Show an editor to allow editing of the bundle metadata.
  -W, --wait            Wait until run finishes.
  -t, --tail            Wait until run finishes, displaying stdout/stderr.
  -v, --verbose         Display verbose output.
"""

parser = argparse.ArgumentParser(description="Run a gcp instance")
parser.add_argument("-f", "--file", type=str, help="path to run yaml")
parser.add_argument("--mangle", action="store_true", help="should we mangle the run name", default=False)
parser.add_argument("-e", "--env", action="append", help="environment variables. either NAME or NAME=VALUE", default=[])
parser.add_argument("-g", "--gpus", type=int, help="number of gpus", default=8)
parser.add_argument("-q", "--queue", type=str, help="queue to run on", default="gcp-small")
parser.add_argument("--disk", "--disk-size", type=str, help="disk size", default=None)
parser.add_argument("--mem", "--memory", type=str, help="memory", default="300g")
parser.add_argument("--cpu", "--cpu-count", type=int, help="cpu count", default=8)
parser.add_argument("-w", "--worksheet", type=str, help="worksheet name", default=None)


args = parser.parse_args()

config = hiyapyco.load(args.file, method=hiyapyco.METHOD_MERGE)
parameters = hiyapyco.load(f"composer/yamls/models/{config['models']}.yaml", method=hiyapyco.METHOD_MERGE)
# override with parameters from the run yaml
parameters.update(config["parameters"])

if args.mangle:
    name = config["name"]
    unique_id = str(uuid.uuid4())[:8]
    name = name + "-" + unique_id
    config["name"] = name

command = config["command"]
# there's something dumb going on with the _n_gpus stuff (which they've removed in newer composer)
# so we'll just do it manually
command = command.replace("{{ parameters['_n_gpus'] }}", str(args.gpus))
# command can have variable interpolation, which is also in the config
# templates look like {{ variable }}, so we need to use a template.
import jinja2
command = jinja2.Template(command).render(**config)

env = []
if os.getenv("WANDB_API_KEY"):
    env.append("WANDB_API_KEY=" + os.getenv("WANDB_API_KEY"))

for e in args.env:
    if "=" in e:
        name, value = e.split("=")
        env.append(f"{name}={value}")
    else:
        env.append(f"{e}={os.getenv(e, '')}")

env = "\n".join(env)
worksheet = args.worksheet
if not worksheet:
    worksheet = subprocess.check_output(['cl', 'work', '-u'], universal_newlines=True).strip()

CONFIG_DIR_NAME = "config"

with tempfile.NamedTemporaryFile(mode="w") as worker_config_f, tempfile.TemporaryDirectory() as jobdir:
    with open(os.path.join(jobdir, "parameters.yaml"), "w") as params_out:
        params_out.write(hiyapyco.dump(parameters))
        params_out.flush()
    with open(os.path.join(jobdir, "command.sh"), "w") as command_out:
        command_out.write("""
#!/bin/bash
MY_DIR=$(dirname "$0")
set -o allexport
source $MY_DIR/env.env
set +o allexport
""")
        command_out.write(command)
        command_out.flush()
    with open(os.path.join(jobdir, "env.env"), "w") as env_out:
        env_out.write(env)
        env_out.flush()

    subprocess.run(["ls", "-l", jobdir])

    print("Uploading temporary directory to codalab...")
    config_bundle = subprocess.check_output(["cl", "up", "-w", worksheet, "-n", CONFIG_DIR_NAME, jobdir]).strip()
    config_bundle = str(config_bundle, "utf-8")

    description= f"""
    name: {config["name"]}
    git_repo: {config["git_repo"]}
    git_branch: {config["git_branch"]} 
    """

    cmd = [
        "cl", "run",
        "-w", worksheet,
        "-n", config["name"],
        "-d", description,
        "--request-docker-image", config["image"],
        "--request-cpus", str(args.cpu),
        "--request-memory", args.mem,
        "--request-gpus", str(args.gpus),
        "--request-queue", args.queue,
        "--request-network",
        f"/mnt/config:{config_bundle}",
        "---",
        "bash", "/mnt/config/command.sh",
    ]

    if args.disk:
        cmd.extend(["--request-disk", args.disk])

    subprocess.run(cmd, check=True)