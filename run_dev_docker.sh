#!/bin/bash
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

cd "$( dirname "${BASH_SOURCE[0]}" )" || exit

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${SCRIPT_DIR}"

USER_NAME=$(id -u -n)
DOCKER_IMG_NAME="intel/xfastertransformer:dev-ubuntu22.04"

TAG="xfastertransformer"
CONTAINER_NAME=${TAG}_$((`date '+%s'`*1000+`date '+%N'`/1000000))

# Check if the current user is root
if [ "$(id -u)" -ne 0 ]; then
    # Docker image name
    DOCKER_IMG_NAME="${TAG}:${USER_NAME}"
    # Handling illegal characters in the docker image names.
    DOCKER_IMG_NAME=$(echo "${DOCKER_IMG_NAME}" | sed -e 's/=/_/g' -e 's/,/-/g')
    # Convert to all lower-case, as per requirement of Docker image names
    DOCKER_IMG_NAME=$(echo "${DOCKER_IMG_NAME}" | tr '[:upper:]' '[:lower:]')

    # Check if the Docker image exists
    if [ -n "$(docker images -q "${DOCKER_IMG_NAME}")" ]; then
        echo "Docker image ${DOCKER_IMG_NAME} exists."
    else
        echo "Docker image ${DOCKER_IMG_NAME} does not exist."
        echo "Build dev image with current USER ID..."

        # Dockerfile to be used in docker build
        DOCKERFILE_PATH="${SCRIPT_DIR}/dockerfiles/Dockerfile.dev"
        DOCKER_CONTEXT_PATH="${SCRIPT_DIR}"

        DOCKER_USER="--build-arg USER_NAME=$(id -u -n) --build-arg USER_UID=$(id -u) \
                --build-arg USER_GROUP=$(id -g -n) --build-arg USER_GID=$(id -g)"

        if [[ ! -f "${DOCKERFILE_PATH}" ]]; then
            die "Invalid Dockerfile path: \"${DOCKERFILE_PATH}\""
        fi

        # Build the docker container.
        echo "Building container (${DOCKER_IMG_NAME})..."
        docker build ${DOCKER_USER} \
            -t "${DOCKER_IMG_NAME}" \
            -f "${DOCKERFILE_PATH}" "${DOCKER_CONTEXT_PATH}"

        # Check docker build status
        if [[ $? != "0" ]]; then
        echo "ERROR: docker build failed. Dockerfile is at ${DOCKERFILE_PATH}"
        fi
    fi
fi

# Print arguments.
echo "WORKSPACE: ${WORKSPACE}"
echo "  (docker container name will be ${CONTAINER_NAME})"
echo ""

# By default the container will be removed once it finish running (--rm)
# and share the PID namespace (--pid=host) so the process inside does not have
# pid 1 and SIGKILL is propagated to the process inside.
docker run -it \
    --rm \
    --privileged=true \
    --pid=host \
    -P \
    --shm-size=16g \
    --name "${CONTAINER_NAME}" \
    -v /data/:/data/ \
    -v "${WORKSPACE}":"${WORKSPACE}" \
    -w "${WORKSPACE}" \
    -e "http_proxy=$http_proxy" \
    -e "https_proxy=$https_proxy" \
    "${DOCKER_IMG_NAME}"
