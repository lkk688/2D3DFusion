#!/usr/bin/env bash
#Copy from https://github.com/lkk688/myROS2/blob/main/scripts/runcontainer.sh

# Run container from image
#./Scripts/runcontainer.sh mycuda11
IMAGE_name=$1 #"myros2:v1"
PLATFORM="$(uname -m)"
echo $PLATFORM

echo "Script executed from: ${PWD}"
BASEDIR=$(dirname $0)
echo "Script location: ${BASEDIR}"

#sudo xhost +si:localuser:root
# Map host's display socket to docker
DOCKER_ARGS+=("-v /tmp/.X11-unix:/tmp/.X11-unix")
DOCKER_ARGS+=("-e DISPLAY")
DOCKER_ARGS+=("-e NVIDIA_VISIBLE_DEVICES=all")
DOCKER_ARGS+=("-e NVIDIA_DRIVER_CAPABILITIES=all")
#DOCKER_ARGS+=("-e FASTRTPS_DEFAULT_PROFILES_FILE=/usr/local/share/middleware_profiles/rtps_udp_profile.xml")

# docker run --runtime nvidia --network host -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix \
#     $CONTAINER_NAME \
#     /bin/bash
docker run -it --rm \
    --privileged \
    --network host \
    ${DOCKER_ARGS[@]} \
    -v /dev/*:/dev/* \
	-v ${PWD}:/work \
    --runtime nvidia \
    --user="admin" \
    --workdir /work \
    $IMAGE_name \
    /bin/bash