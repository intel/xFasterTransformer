FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc-12 \
    g++-12 \
    make \
    vim \
    wget \
    libnuma-dev \
    numactl \
    git \
    libjsoncpp-dev \
    pkg-config \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 60 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 60


# Install python
RUN apt install software-properties-common -y \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt update -y \
    && apt install python3.8 python3.8-dev python3.8-distutils python3.8-venv -y \
    && wget https://bootstrap.pypa.io/get-pip.py \
    && python3.8 get-pip.py \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.8 60 \
    && update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.8 60 \
    && python -m pip install --upgrade pip \
    && rm get-pip.py

RUN pip install cmake==3.26.1 transformers==4.28.1 sentencepiece==0.1.99 tokenizers==0.13.3
RUN pip install torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Install oneCCL
RUN git clone https://github.com/oneapi-src/oneCCL.git /tmp/oneCCL \
    && cd /tmp/oneCCL \
    && git checkout 2021.9 \
    && sed -i 's/cpu_gpu_dpcpp/./g' cmake/templates/oneCCLConfig.cmake.in \
    && mkdir build \
    && cd build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local/oneCCL \
    && make -j install \
    && cd ~ \
    && rm -rf /tmp/oneCCL
    
RUN echo "source /usr/local/oneCCL/env/setvars.sh" >> ~/.bashrc

WORKDIR /usr/local/

RUN wget https://github.com/oneapi-src/oneDNN/releases/download/v0.21/mklml_lnx_2019.0.5.20190502.tgz \
    && tar -xzf mklml_lnx_2019.0.5.20190502.tgz \
    && rm -f mklml_lnx_2019.0.5.20190502.tgz

RUN echo 'export LD_LIBRARY_PATH=/usr/local/mklml_lnx_2019.0.5.20190502/lib:$LD_LIBRARY_PATH' >> /root/.bashrc

WORKDIR /root/