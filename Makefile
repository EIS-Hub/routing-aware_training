# Define the name of your conda environment here
ENV_NAME=.mosaic_env

# Path to the Conda executable
CONDA=conda

# Default target executed when no arguments are given to make.
default: create install

install_conda : 
	@mkdir -p ~/miniconda3
	@wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
	@bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
	@rm -rf ~/miniconda3/miniconda.sh
	~/miniconda3/bin/conda init bash

# Target to create a new conda environment from the environment.yml file
create:
	@echo "Creating the conda environment..."
	$(CONDA) create --name $(ENV_NAME) python=3.12 -y

# Target to install additional Python packages from a list in a file
install: create
	@echo "Installing additional packages from py_req.txt into the conda environment..."
	$(CONDA) run -n $(ENV_NAME) pip install --upgrade pip
	$(CONDA) run -n $(ENV_NAME) pip install -r py_req.txt
	$(CONDA) run -n $(ENV_NAME) pip install --upgrade optax==0.2.2 jax==0.4.25 jaxlib==0.4.25+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
	$(CONDA) run -n $(ENV_NAME) pip install torch --index-url https://download.pytorch.org/whl/cu118

.PHONY: create install
all: install

# Target to delete the conda environment
clean:
	@echo "Removing the conda environment..."
	$(CONDA) env remove -n $(ENV_NAME)

.PHONY: default create install 
