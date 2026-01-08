# Variables
SOURCE_IMAGE ?= docker.io/library/adeo-omniasr-server
# Default repository path: platform/mlops/mlops-serving/hamsa-stt-onprem-v2
# User can override this: make REPOSITORY=my/custom/path ...
REPOSITORY ?= platform/mlops/mlops-serving/adeo-omniasr-server
TAG := $(shell date +%Y%m%d)
REGISTRY ?= harbor.adeoaiengine.ecouncil.ae/openinnovationai

# Full target name for registry
FULL_TARGET := $(REGISTRY)/$(REPOSITORY):$(TAG)

.PHONY: default harbor_upload

default:
	@echo "Tagging $(SOURCE_IMAGE) as $(REPOSITORY):$(TAG) (Local placeholder)..."
	# We tag it locally as the full target so it is ready to push, 
	# or we can just keep the local tag simple. 
	# The user requested "docker tag SOURCE_IMAGE[:TAG] registry.../REPOSITORY[:TAG]"
	# so we will do that in harbor_upload, but we can also do it here for convenience.
	docker tag $(SOURCE_IMAGE) $(FULL_TARGET)
	@echo "Done. Created tag: $(FULL_TARGET)"

harbor_upload:
	@echo "Tagging $(SOURCE_IMAGE) for registry..."
	podman tag $(SOURCE_IMAGE) $(FULL_TARGET)
	@echo "Pushing $(FULL_TARGET) to registry..."
	podman push --tls-verify=false $(FULL_TARGET)
	@echo "Done."


## podman login --tls-verify=false harbor.adeoaiengine.ecouncil.ae
## login: custom-images
## password: CustomImages123