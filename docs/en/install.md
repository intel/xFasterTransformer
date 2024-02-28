# Installation
## From PyPI
It is recommended to install the CPU version of Torch first, as the default version of Torch includes CUDA components, resulting in a larger installation package size.
```bash
pip install torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu

pip install xfastertransformer
```

## Using Docker
```bash
docker pull intel/xfastertransformer:latest
```
Run the docker with the command (Assume model files are in `/data/` directory):  
```bash
docker run -it \
    --name xfastertransformer \
    --privileged \
    --shm-size=16g \
    -v /data/:/data/ \
    -e "http_proxy=$http_proxy" \
    -e "https_proxy=$https_proxy" \
    intel/xfastertransformer:latest
```
**Notice!!!**: Please enlarge `--shm-size` if  **bus error** occurred while running in the multi-ranks mode . The default docker limits the shared memory size to 64MB and our implementation uses many shared memories to achieve a  better performance.

## Built from source
### Prepare Environment
#### Manually
- [PyTorch](https://pytorch.org/get-started/locally/) v2.0 (When using the PyTorch API, it's required, but it's not needed when using the C++ API.)
  ```bash 
  pip install torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu
  ```

#### How to build
- Using 'CMake'
  ```bash
  # Build xFasterTransformer
  git clone https://github.com/intel/xFasterTransformer.git xFasterTransformer
  cd xFasterTransformer
  git checkout <latest-tag>
  # Please make sure torch is installed when run python example
  mkdir build && cd build
  cmake ..
  make -j
  ```

- Using `python setup.py`
  ```bash
  # Build xFasterTransformer library and C++ example.
  python setup.py build

  # Install xFasterTransformer into pip environment.
  # Notice: Run `python setup.py build` before installation!
  python setup.py install