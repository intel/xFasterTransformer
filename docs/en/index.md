<h1 align="center">
    xFasterTransformer
</h1>

# xFasterTransformer

xFasterTransformer is an exceptionally optimized solution for large language models (LLM) on the X86 platform, which is similar to FasterTransformer on the GPU platform. xFasterTransformer is able to operate in distributed mode across multiple sockets and nodes to support inference on larger models. Additionally, it provides both C++ and Python APIs, spanning from high-level to low-level interfaces, making it easy to adopt and integrate.

# Getting Started

```{toctree}
:maxdepth: 2
:caption: Install

install
```

# Feature

```{toctree}
:maxdepth: 2
:caption: Optimization

flash_attention
```

# Development Guide

```{toctree}
:maxdepth: 2
:caption: Expanding xFasterTransformer

dev_guide
```

# Distributed Deployment

```{toctree}
:maxdepth: 2
:caption: Demo with eRDMA using QWen-72B

dist_demo
```