FROM nvcr.io/nvidia/pytorch:22.02-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
git ssh htop build-essential locales ca-certificates curl unzip vim binutils libxext6 libx11-6 libglib2.0-0 \
libxrender1 libxtst6 libxi6 tmux screen nano wget gcc python3-dev python3-setuptools python3-venv ninja-build sudo apt-utils less\
    zstd libzstd-dev wget

RUN sudo apt-get autoclean && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# codalab ignores whatever user you make so
#RUN useradd --create-home --uid 1000 --shell /bin/bash mosaic
#RUN usermod -aG sudo mosaic
#RUN echo "mosaic ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
## # Change to non-root privilege
#USER mosaic

# install a bundle of dependencies that composer installs anyway
#ENV PATH $PATH:/home/mosaic/.local/bin

RUN pip install --no-cache-dir transformers "datasets<2" 'braceexpand' 'zstandard' 'fsspec' "pyyaml>=5.4.1,<6" "tqdm>=4.62.3,<5" "torchmetrics>=0.7.0,<0.8" \
    "torch_optimizer>=0.1.0,<0.2" "torch>=1.11,<2" "yahp==0.1.0" "requests>=2.26.0,<3" "numpy>=1.21.5,<2"  "apache-libcloud>=3.3.1,<4" \
    "psutil>=5.8.0,<6" "coolname>=1.1.0,<2" "wandb>=0.12.10,<0.13" "wurlitzer>=3.0.2,<4" deepspeed \
    "custom_inherit==2.3.2" "junitparser==2.4.3" "coverage[toml]==6.3.2" "fasteners==0.17.3" "pytest==7.1.0" \
    "toml==0.10.2" "yapf==0.32.0" "isort==5.10.1" "ipython==7.32.0" "ipykernel==6.9.2" "jupyter==1.0.0" "yamllint==1.26.3" \
    "pytest-timeout==2.1.0" "recommonmark==0.7.1" "sphinx==4.4.0" "docutils==0.17.1" "sphinx_markdown_tables==0.0.15" \
    "sphinx-argparse==0.3.1" "sphinxcontrib.katex==0.8.6" "sphinxext.opengraph==0.6.1" "sphinxemoji==0.2.0" "furo==2022.3.4" \
    "sphinx-copybutton==0.5.0" "testbook==0.4.2" "myst-parser==0.16.1" "pylint==2.12.2" "docformatter==1.4" "sphinx_panels==0.6.0" \
    "sphinxcontrib-images==0.9.4" "webdataset==0.1.103" "monai>=0.8.0,<0.9" "scikit-learn>=1.0.1,<2" "vit_pytorch==0.27" \
     --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip install --no-cache-dir triton==1.0.0 timm==0.5.4 torchdata "pycocotools>=2.0.4"

ENV PATH $PATH:/root/.local/bin
