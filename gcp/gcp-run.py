# launches a gcp instance with a docker image and runs a run yaml a la mcli run
import argparse
import os

import hiyapyco
import uuid
import tempfile

# gcp-run.py -f my-yaml

parser = argparse.ArgumentParser(description="Run a gcp instance")
parser.add_argument("-f", "--file", type=str, help="path to run yaml")
parser.add_argument("--mangle", action="store_true", help="should we mangle the run name", default=True)
parser.add_argument("-g", "--gpus", type=int, help="number of gpus", default=2)
parser.add_argument("-a", "--accelerator", type=str, help="accelerator type", default="a100")
parser.add_argument("-z", "--zone", type=str, help="zone", default="us-central1-c")
parser.add_argument("--machine-type", type=str, help="machine type", default="n1-standard-32")
parser.add_argument("-e", "--env", action="append", help="environment variables. either NAME or NAME=VALUE", default=[])

args = parser.parse_args()

A1_MACHINE_TYPES = {
    1: "a2-highgpu-1g",
    2: "a2-highgpu-2g",
    4: "a2-highgpu-4g",
    8: "a2-highgpu-8g",
    16: "a2-megagpu-16g",
}

if "a100" in args.accelerator:
    if args.gpus not in A1_MACHINE_TYPES:
        print("ERROR: Accelerator count not supported. Supported counts are: {}".format(A1_MACHINE_TYPES.keys()))
        exit(1)
    machine_type = A1_MACHINE_TYPES[args.gpus]
    if machine_type != args.machine_type:
        print("WARNING: machine type {} not supported for accelerator count {} with a100s. Using {} instead".format(
            args.machine_type, args.gpus, machine_type))
        args.machine_type = machine_type

base_config = f"""
"""

config = hiyapyco.load([base_config, args.file], method=hiyapyco.METHOD_MERGE)
parameters = hiyapyco.load(f"composer/yamls/models/{config['models']}.yaml", method=hiyapyco.METHOD_MERGE)
# override with parameters from the run yaml
parameters.update(config["parameters"])

# print(config)
# print(parameters)

docker_image = config["image"]
if ".io" not in docker_image:
    docker_image = f"docker.io/{docker_image}"

name = config["name"]
if args.mangle:
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

if False: # vertex ai pricing is stupid
    with tempfile.NamedTemporaryFile(mode="w") as worker_config_f, tempfile.TemporaryDirectory() as jobdir:
        with open(os.path.join(jobdir, "parameters.yaml"), "w") as params_out:
            params_out.write(hiyapyco.dump(parameters, method=hiyapyco.METHOD_MERGE))

        with open(os.path.join(jobdir, "command.sh"), "w") as command_out:
            command_out.write(command)

        worker_config = f"""
        workerPoolSpecs:
          machineSpec:
            machineType: {args.machine_type}
            acceleratorType: {args.accelerator}
            acceleratorCount: {args.gpus}
          local-package-path: 
          replicaCount: 1
          containerSpec:
            imageUri: {docker_image}
            env:
              - name: WANDB_API_KEY
              - value: {os.getenv("WANDB_API_KEY", "")}
        """

        worker_config_f.write(worker_config)
        worker_config_f.flush()


        import subprocess

        args = [
            "gcloud",
            "ai",
            "custom-jobs",
            "create",
            f"--display-name={name}",
            f"--config={worker_config_f.name}",
        ]

    # # want to invoke the args, using subprocess
    # import subprocess
    # import shlex
    # print(shlex.join(args))
    # process = subprocess.Popen(args, stdin=subprocess.PIPE,
    #                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # stdout, stderr = process.communicate()
    # stdout.splitlines()
else:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh") as startup_command_out:
        host_config_dir = "/tmp/config"
        script_template = open("gcp/gcp-startup-script-template.sh").read()
        script = jinja2.Template(script_template).render(command=command, parameters=hiyapyco.dump(parameters), env=env)
        # print(script)
        startup_command_out.write(script)
        startup_command_out.flush()


        # instead we'll use gcloud gce, which are more annoying to work with, but are preemptible (and a lot cheaper)
        # commands look like this:
            # gcloud compute instances create-with-container mistral-sprucfluo-gpt2-11m-a4bd07ew3 --container-image mosaicml/pytorch --container-command 'git clone https://github.com/stanford-crfm/composer $HOME/composer
            # cd $HOME/composer
            # echo '"'"'Checking out composer branch fancy_dataset'"'"'
            # git checkout fancy_dataset
            # pip3 install torch==1.11 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
            # pip install --user -e .[all]
            # composer -n 4 examples/run_composer_trainer.py -f /mnt/config/parameters.yaml' --zone us-central1-a --accelerator type=nvidia-tesla-v100,count=4 --machine-type n1-standard-32 --maintenance-policy TERMINATE --restart-on-failure --image-family cos-85-lts --image-project cos-cloud --boot-disk-size 500GB --metadata-from-file user-data=gcp/cloud-init.yaml --preemptible --verbosity debug"
        # the main annoyance is that gce doesn't exit when the container finishes, so we have to rely on
        # https://stackoverflow.com/a/58215421/1736826 to make it exit
        # gcloud compute --project=PROJECT_NAME instances create INSTANCE_NAME  \
        # --zone=ZONE --machine-type=TYPE \
        # --metadata=image_name=IMAGE_NAME,\
        # container_param="PARAM1 PARAM2 PARAM3",\
        # startup-script-url=PUBLIC_SCRIPT_URL \
        # --maintenance-policy=MIGRATE --service-account=SERVICE_ACCUNT \
        # --scopes=https://www.googleapis.com/auth/cloud-platform --image-family=cos-stable \
        # --image-project=cos-cloud --boot-disk-size=10GB --boot-disk-device-name=DISK_NAME
        cmd = [
            "gcloud", "compute", "instances",
            "create", name,
            "--zone", args.zone,
            "--machine-type", args.machine_type,
            "--service-account=instance-deleter@hai-gcp-models.iam.gserviceaccount.com",
            "--scopes=https://www.googleapis.com/auth/cloud-platform",
            "--metadata", f"install-nvidia-driver=True,image_name={docker_image}",
            "--metadata-from-file", f"user-data=gcp/cloud-init.yaml",
            f"--metadata-from-file=startup-script={startup_command_out.name}",
            "--maintenance-policy", "TERMINATE",
            "--restart-on-failure",
            "--image-project", "deeplearning-platform-release",
            "--image-family", "common-cu113",
            "--boot-disk-size", "500GB",
            # "--preemptible",
            "--provisioning-model=SPOT",
            "--verbosity", "debug",
        ]

        if args.machine_type not in A1_MACHINE_TYPES.values():
            cmd += ["--accelerator", f"type={args.accelerator},count={args.gpus}"]

        import subprocess
        import shlex
        print("About to run:")
        print()
        print(">>>", shlex.join(cmd))
        print()
        print("*** This could take a while, please be patient and will write very little ***")
        subprocess.run(cmd)
