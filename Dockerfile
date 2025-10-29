FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
# FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
# Install essential Linux packages
RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl 

WORKDIR /app/

RUN mkdir -p /data2/

ARG CACHE_BUSTER=1

# RUN git clone https://github.com/raja-kumar/Visual-RFT

# COPY ./ /app/DiVE-k/

WORKDIR /app/DiVE-k

RUN bash setup.sh

RUN pip install flash-attn --no-build-isolation && pip install deepspeed==0.16.3

RUN pip install trl==0.16.0 && pip install json_repair && pip install matplotlib && pip install peft
