FROM gcr.io/kaggle-gpu-images/python:latest

WORKDIR /workspace

RUN apt update
RUN apt install -y \
    git \
    bash-completion \
    python3-llvmlite \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev

RUN echo "source /etc/bash_completion" >> /root/.bashrc

RUN pip install -U pip
RUN pip install black isort mypy pyproject-flake8
RUN pip install pytest-cov
RUN pip install iterative-stratification hydra-core
