VERSION ?= latest

.PHONY: build push

build: Dockerfile
    docker build --no-cache -t vatevr/cybathlon-recordings:$(VERSION) .

push:
    docker push vatevr/cybathlon-recordings:$(VERSION)


default: build
