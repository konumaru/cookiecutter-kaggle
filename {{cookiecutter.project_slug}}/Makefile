default: help

.PHONY: help
help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done

.PHONY: init
init: # Initialize the project.
	@echo "Initializing project..."
	cp .envrc.example .envrc

.PHONY: dl
dl: # Download the data and initialize the project.
	@echo "Downloading data..."
	./bin/download.sh

.PHONY: tests
tests: # Run tests
	@echo "Running tests..."
	pytest --cov -v tests/

.PHONY: train
train: # Train the model
	@echo "Training model..."
	./bin/train.sh

.PHONY: submit
submit: # Submit the model to kaggle.
	@echo "Submitting model..."
	# $(MAKE) test
	./bin/upload.sh
	./bin/submit.sh
