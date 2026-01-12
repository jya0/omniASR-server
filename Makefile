# Variables
BASE_IMAGE := localhost/adeo-omniasr-base
APP_IMAGE := localhost/adeo-omniasr-server
REPOSITORY ?= platform/mlops/mlops-serving/adeo-omniasr-server
TAG := $(shell date +%Y%m%d)
REGISTRY ?= harbor.adeoaiengine.ecouncil.ae/openinnovationai

# Full target name for registry
FULL_TARGET := $(REGISTRY)/$(REPOSITORY):$(TAG)

.PHONY: base build run stop logs clean harbor_upload default

# Build base image (slow, contains models - run once)
base:
	@echo "Building base image $(BASE_IMAGE)..."
	podman build -f Dockerfile.base -t $(BASE_IMAGE) .
	@echo "Done. Base image ready: $(BASE_IMAGE)"

# Build app image (fast, uses cached base)
build:
	@echo "Building app image $(APP_IMAGE)..."
	podman build -t $(APP_IMAGE) .
	@echo "Done. App image ready: $(APP_IMAGE)"

# Build everything (base + app)
all: base build

# Run the container with GPU
run:
	@echo "Running $(APP_IMAGE)..."
	podman run -d --rm \
		--name omniasr-server \
		--device nvidia.com/gpu=all \
		-p 8080:8080 \
		$(APP_IMAGE)
	@echo "Server running at http://localhost:8080"

# Run with GPU (alternative syntax for older Podman versions)
run-gpu:
	@echo "Running $(APP_IMAGE) with GPU..."
	podman run -d --rm \
		--name omniasr-server \
		--security-opt=label=disable \
		--hooks-dir=/usr/share/containers/oci/hooks.d/ \
		-p 8080:8080 \
		$(APP_IMAGE)
	@echo "Server running at http://localhost:8080"

# Run in foreground (see logs live in terminal/VSCode)
run-fg:
	@echo "Running $(APP_IMAGE) in foreground..."
	podman run --rm \
		--name omniasr-server \
		--security-opt=label=disable \
		--hooks-dir=/usr/share/containers/oci/hooks.d/ \
		-p 8080:8080 \
		$(APP_IMAGE)

# Run without GPU (CPU only)
run-cpu:
	@echo "Running $(APP_IMAGE) in CPU mode..."
	podman run -d --rm \
		--name omniasr-server \
		-e DEVICE=cpu \
		-p 8080:8080 \
		$(APP_IMAGE)
	@echo "Server running at http://localhost:8080"

# Stop the container
stop:
	@echo "Stopping omniasr-server..."
	podman stop omniasr-server || true

# View logs
logs:
	podman logs -f omniasr-server

# Clean up images
clean:
	@echo "Removing images..."
	podman rmi $(APP_IMAGE) || true
	podman rmi $(BASE_IMAGE) || true

# Tag for local reference
default:
	@echo "Tagging $(APP_IMAGE) as $(FULL_TARGET)..."
	podman tag $(APP_IMAGE) $(FULL_TARGET)
	@echo "Done. Created tag: $(FULL_TARGET)"

# Upload to Harbor registry
harbor_upload:
	@echo "Tagging $(APP_IMAGE) for registry..."
	podman tag $(APP_IMAGE) $(FULL_TARGET)
	@echo "Pushing $(FULL_TARGET) to registry..."
	podman push --tls-verify=false $(FULL_TARGET)
	@echo "Done."

# Login to Harbor (run once)
# podman login --tls-verify=false harbor.adeoaiengine.ecouncil.ae
# login: custom-images
# password: CustomImages123