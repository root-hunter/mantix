#include "mantix/mantix.h"

#include <inttypes.h>
#include <stdio.h>

int main(void)
{
    mtx_float value;
    mtx_status status = mtx_init(&value, 256U);

    if (status == MTX_OK) {
        status = mtx_set_u64(&value, UINT64_C(40));
    }
    if (status != MTX_OK) {
        fprintf(stderr, "mantix: %s\n", mtx_status_string(status));
        mtx_clear(&value);
        return 1;
    }

    printf("40 = %" PRIu64 " * 2^%" PRId64 " (%zu-bit target precision)\n",
           value.limbs[0], value.exponent, value.precision);
    mtx_clear(&value);
    return 0;
}
