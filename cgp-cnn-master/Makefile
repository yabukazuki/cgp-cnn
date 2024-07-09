# --shm-size オプションの必要性の検討
IMAGE_NAME := yabukazuki/cgp-cnn:0.1.0
CONTAINER_NAME := cgp-cnn.yabuuchi

# docker commit cgp-cnn.yabuuchi yabukazuki/cgp-cnn:0.1.0

.PHONY: build
build:
	@docker build . -t $(IMAGE_NAME)

.PHONY: run
run:
	@docker run --rm -it \
        -u root \
        --gpus all \
        --name $(CONTAINER_NAME) \
        -v $(shell pwd):/var/www \
        $(IMAGE_NAME) bash
