FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
# FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
# Install essential Linux packages
RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl 

WORKDIR /app/

RUN mkdir -p /data2/
RUN mkdir -p /app/DiVE-k

ARG CACHE_BUSTER=1

# RUN git clone https://github.com/raja-kumar/DiVE-k

COPY ./ /app/DiVE-k

WORKDIR /app/DiVE-k

RUN bash setup.sh

RUN pip install deepspeed==0.16.3

RUN pip install trl==0.16.0 && pip install json_repair && pip install matplotlib && pip install peft==0.17.1 && pip install math_verify && pip install --upgrade wandb

RUN pip install git+https://github.com/huggingface/transformers.git@8ee50537fe7613b87881cd043a85971c85e99519

RUN pip install flash-attn==2.7.3 --no-build-isolation
