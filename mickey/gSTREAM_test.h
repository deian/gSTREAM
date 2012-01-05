#ifndef __GSTREAM_TEST_H_
#define __GSTREAM_TEST_H_

#include <stdlib.h>
#include "gSTREAM.h"

#define PRINT_KEYIV  0
#define PRINT_INPUT  0
#define PRINT_OUTPUT 1

/* Given a seed, the number of streams=nr_threads*nr_blocks, key size,
   iv size, and buffer byte-size generate buff_size_bytes keystream bytes.
   If action is ENCRYPT or DECRYPT, a random buffer of same size is 
   created and the respective action (rather than generate keystream)
   is performed.
*/

void do_test(int seed, int dev_no, int nr_threads, int nr_blocks,
             gSTREAM_action action,
             size_t key_size_bytes,
             size_t iv_size_bytes, size_t buff_size_bytes);

#endif
