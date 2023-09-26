#pragma once
#include <tuple>

#include "messenger.h"
#include "transformer_ctx.h"

class AbstractDecoder {
public:
    // Forward function with the input IDs with shape of dims - (batchSize, beamSize, seqLen)
    // Return the decoding result, split offset, and split size
    // The returned result is a split representing the possibilities of next token, like the shadow part in below graph
    //                                         splitOffset
    //                                              \ 
    //                                               \|<-splitSize->|
    //    _                ___________________________v______________________________________
    //    ^               |             |             |||||||||||||||             |          |
    //    |               |             |             |||||||||||||||             |          |
    // batchSize*beamSize |             |             |||||||||||||||             |          |
    //    |               |             |             |||||||||||||||             |          |
    //    v               |_____________|_____________|||||||||||||||_____________|__________|
    //                    |<----------------------- vocabSize  ----------------------------->|
    virtual std::tuple<float *, int, int> forward(int *ids, int64_t *dims, int step, bool logits_all = false) = 0;

    // Reorder cached keys and values, size=batchSize*beamSize
    virtual void reorderCache(int *idx, int size) = 0;

    virtual DecoderContext *getContext() = 0;

    virtual Messenger &getMessenger() = 0;

    virtual int getRank() = 0;

    virtual int getEndId() = 0;
};
