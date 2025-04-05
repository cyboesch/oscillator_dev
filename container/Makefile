.PHONY: docker-build docker-build-cpu docker-run docker-run-cpu docker-push build-apptainer

IMAGE_NAME := docker_thermo_ai
PROJECT_SPACE := gen-osc
IONIC_USER := cb7454
DOCKER_USER := cb7454

docker-build:
	docker build --tag ${IMAGE_NAME}:latest .

docker-build-cpu:
	docker build --tag ${IMAGE_NAME}_cpu:latest --file ./build/Dockerfile.cpu .

docker-run:
	docker run --gpus all -dt -v "$(shell pwd)/../":"/workspace" --name ${IMAGE_NAME} ${IMAGE_NAME}:latest /bin/bash

docker-run-MNISTlearning-gpu:
	docker exec -it ${IMAGE_NAME} python3 projects/physical-diffusion-models/scripts/duffing_network_learning_MNIST_sde.py

docker-run-cpu:
	docker run  -dt -v "$(shell pwd)":"/${IMAGE_NAME}" --name ${IMAGE_NAME}_cpu ${IMAGE_NAME}_cpu:latest /bin/bash


build-apptainer: docker-build
	apptainer build ${IMAGE_NAME}.sif docker-daemon://${IMAGE_NAME}:latest
	scp ${IMAGE_NAME}.sif ${IONIC_USER}@soak.cs.princeton.edu:/n/fs/${PROJECT_SPACE}/