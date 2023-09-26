#pragma once
#include <cstdlib>

extern "C" {
// Allocate memory on preferred node, if the preferred node has been set
// If the preferred node is not set, allocate on local
void *xft_numa_alloc(size_t size);

// Allocate memory on a specified node
void *xft_numa_alloc_onnode(size_t size, int node);

// Free the memory allocated by us
void xft_numa_free(void *start, size_t size);

// Set preferred node to allocate memory
void xft_set_preferred_node(int node);
}