# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
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
    pkg-config \
    software-properties-common \
    zlib1g-dev \
    libssl-dev \
    libffi-dev \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 60 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 60 \
    && rm -rf /var/lib/apt/lists/*


# Install python
WORKDIR /tmp
RUN wget -q https://www.python.org/ftp/python/3.8.10/Python-3.8.10.tgz \
    && tar -xzvf Python-3.8.10.tgz

WORKDIR /tmp/Python-3.8.10
RUN ./configure --prefix=/usr/bin/python3.8 --enable-optimizations \
    && make -j \
    && make install \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.8/bin/python3.8 60 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/python3.8/bin/pip3 60 \
    && python -m pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir wheel \
    && rm -rf /tmp/*

RUN pip install --no-cache-dir torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir cmake==3.26.1 transformers==4.30.0 sentencepiece==0.1.99 tokenizers==0.13.3 accelerate==0.23.0 \
    && ln -s /usr/bin/python3.8/lib/python3.8/site-packages/cmake/data/bin/cmake /usr/bin/cmake

# Install oneCCL
RUN git clone https://github.com/oneapi-src/oneCCL.git /tmp/oneCCL

WORKDIR /tmp/oneCCL
RUN git checkout 2021.10 \
    && sed -i 's/cpu_gpu_dpcpp/./g' cmake/templates/oneCCLConfig.cmake.in \
    && mkdir build
WORKDIR /tmp/oneCCL/build
RUN cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local/oneCCL \
    && make -j install 

RUN echo "source /usr/local/oneCCL/env/setvars.sh" >> ~/.bashrc

WORKDIR /usr/local/
RUN rm -rf /tmp/oneCCL

RUN wget -q https://github.com/oneapi-src/oneDNN/releases/download/v0.21/mklml_lnx_2019.0.5.20190502.tgz \
    && tar -xzf mklml_lnx_2019.0.5.20190502.tgz \
    && rm -f mklml_lnx_2019.0.5.20190502.tgz

RUN echo "export LD_LIBRARY_PATH=/usr/local/mklml_lnx_2019.0.5.20190502/lib:\$LD_LIBRARY_PATH" >> /root/.bashrc

WORKDIR /root/