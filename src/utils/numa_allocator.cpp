#include "numa_allocator.h"
#include <cstdio>
#include <numa.h>

static int preferredNode = -1;

void *xft_numa_alloc(size_t size) {
    return xft_numa_alloc_onnode(size, preferredNode);
}

void *xft_numa_alloc_onnode(size_t size, int node) {
    void *memory = nullptr;

    if (node >= 0) {
        memory = numa_alloc_onnode(size, node);
    } else {
        memory = numa_alloc(size);
    }

    if (memory == nullptr) {
        printf("Failed to allocate memory (size=%zu, node=%d)\n", size, node);
        exit(-1);
    }

    return memory;
}

void xft_numa_free(void *start, size_t size) {
    numa_free(start, size);
}

void xft_set_preferred_node(int node) {
    preferredNode = node;
}