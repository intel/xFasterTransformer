# 安装
## 从 PyPI 安装
建议先安装CPU版本的PyTorch，因为默认版本的PyTorch包含CUDA组件，安装包体积较大。
```bash
pip install torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu

pip install xfastertransformer
```

## 使用Docker
```bash
docker pull intel/xfastertransformer:latest
```
使用以下命令运行 Docker（此处假设模型文件位于`/data/`目录下）：
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

**注意！！！**：如果在分布式模式下运行时出现**bus error**，请增大`--shm-size`。Docker默认限制共享内存大小为64MB，而在xFasterTransformer的实现中会使用较多的共享内存来获得更好的性能。

## 从源代码构建
### 准备环境
#### 手动准备
- [PyTorch](https://pytorch.org/get-started/locally/) v2.0 (这在使用Python API时是必需的，但如果使用C++ API则不需要安装)
  ```bash 
  pip install torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu
  ```

#### 构建步骤
- Using 'CMake'
  ```bash
  # 构建xFasterTransformer
  git clone https://github.com/intel/xFasterTransformer.git xFasterTransformer
  cd xFasterTransformer
  git checkout <latest-tag>
  # 使用Python API时请确保已安装PyTorch
  mkdir build && cd build
  cmake ..
  make -j
  ```

- Using `python setup.py`
  ```bash
  # 构建 xFasterTransformer 库和 C++ 示例。
  python setup.py build

  # 将 xFasterTransformer 安装到 pip 环境中。
  # 注意：在安装之前请运行 `python setup.py build`！
  python setup.py install