<h1 align="center">
    xFasterTransformer
</h1>

# xFasterTransformer

xFasterTransformer为大语言模型（LLM）在CPU X86平台上的部署提供了一种深度优化的解决方案，支持多CPU节点之间的分布式部署方案，使得超大模型在CPU上的部署成为可能。此外，xFasterTransformer提供了C++和Python两种API接口，涵盖了从上层到底层的接口调用，易于用户使用并将xFasterTransformer集成到自有业务框架中。

# 开始

```{toctree}
:maxdepth: 2
:caption: 安装

install
```

# 功能

```{toctree}
:maxdepth: 2
:caption: 优化介绍

flash_attention
```


# 分布式部署

```{toctree}
:maxdepth: 2
:caption: 使用eRDMA部署QWen-72B模型样例

dist_demo
```