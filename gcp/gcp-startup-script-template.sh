#!/bin/bash

sudo cos-extensions install gpu

while ! [[ -x "$(command -v nvidia-smi)" ]];
do
  echo "sleep to check"
  sleep 5s
done
echo "nvidia-smi is installed"


# get image name and container parameters from the metadata
IMAGE_NAME=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/image_name -H "Metadata-Flavor: Google")

CONTAINER_PARAM=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/container_param -H "Metadata-Flavor: Google")


export CONFIG_DIR=/tmp/config

mkdir $CONFIG_DIR

cat <<\EOF > $CONFIG_DIR/run.sh
{{ command }}
EOF

chmod +x $CONFIG_DIR/run.sh

cat <<\EOF > $CONFIG_DIR/parameters.yaml
{{ parameters }}
EOF

cat <<\EOF > $CONFIG_DIR/env.env
{{ env }}
EOF


## This is needed if you are using a private images in GCP Container Registry
## (possibly also for the gcp log driver?)
sudo HOME=/home/root /usr/bin/docker-credential-gcr configure-docker

curl --silent --connect-timeout 1 -f -H "Metadata-Flavor: Google" http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/scopes

gcloud auth configure-docker

start_docker () {
  echo "Docker run with GPUs"
  docker run --env-file $CONFIG_DIR/env.env --log-driver=gcplogs --gpus all --mount type=bind,source=$CONFIG_DIR,target=/mnt/config $IMAGE_NAME bash /mnt/config/run.sh
  EXIT_CODE=$?
  echo "Docker finished. Exit Code is $EXIT_CODE"
  return $EXIT_CODE
}

start_docker


if [[ $EXIT_CODE -eq 0 ]]; then
  echo "Successfully finished"
else
  echo "Docker failed. Exit Code is $EXIT_CODE. Trying one more time"
  start_docker
fi

if [[ $EXIT_CODE -eq 0 ]]; then
  echo "Successfully finished"
else
  echo "Docker failed again. Exit Code is $EXIT_CODE. Giving up. Sleeping 2h before killing the instance"
  sleep 2h
fi

echo "Kill VM $(hostname)"
gcloud compute instances delete $(hostname) --zone \
"$(curl -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone -s | cut -d/ -f4)" -q

