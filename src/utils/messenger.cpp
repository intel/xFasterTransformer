#include "messenger.h"

Messenger *gmessenger;

void globalBarrier() {
    gmessenger->barrier();
}