#!/bin/bash

sudo cos-extensions install gpu

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


## This is needed if you are using a private images in GCP Container Registry
## (possibly also for the gcp log driver?)
#sudo HOME=/home/root /usr/bin/docker-credential-gcr configure-docker

# Run! The logs will go to stack driver
#sudo HOME=/home/root  docker run --log-driver=gcplogs ${IMAGE_NAME} ${CONTAINER_PARAM}

# Get the zone
zoneMetadata=$(curl "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor:Google")
# Split on / and get the 4th element to get the actual zone name
IFS=$'/'
zoneMetadataSplit=($zoneMetadata)
ZONE="${zoneMetadataSplit[3]}"

# Run compute delete on the current instance. Need to run in a container
# because COS machines don't come with gcloud installed
docker run --entrypoint "gcloud" google/cloud-sdk:alpine compute instances delete ${HOSTNAME}  --delete-disks=all --zone=${ZONE}
