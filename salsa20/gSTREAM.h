#ifndef __GSTREAM_H_
#define __GSTREAM_H_

#define DEBUG
#include <stdint.h>

#ifdef DEBUG
#define debug(...)                                              \
   fprintf(stderr, __VA_ARGS__)
#else
#define debug(...) ;
#endif

#define CH_ENDIANESS32(a) (a)


typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef enum { ENCRYPT=0, DECRYPT=1, GEN_KEYSTREAM=2 } gSTREAM_action;

typedef struct {
   u32 *x_d;
} Salsa20_ctx;

typedef struct {
   int nr_threads; /* per block */
   int nr_blocks;

   u32 *keys_d;
   u32 key_size; /* in bits */
   int allocated_keys;

   u32 *ivs_d;
   u32 iv_size; /* in bits */
   int allocated_ivs;

   u32 *buff_d, *buff_h;
   u32 buff_size; /* in bytes (ceil to nearest 4-bytes) */
   int allocated_buff;

   struct { /* expandable benchmarking struct */
      unsigned timer;
   } bench; 

   /* Insert cipher-dependent fields here: */
   Salsa20_ctx sctx;

} gSTREAM_ctx;

/* Initialize device and allocate any state-related buffers.
   device - which device to use,
   nr_threads - number of threads/block,
   nr_blocks - number of blocks/grid
*/
void gSTREAM_init(gSTREAM_ctx* ctx, int device, int nr_threads, int nr_blocks);

/* Do the key setup.
   keys - all the stream keys: key[i][] corresponds to the i-th streams's key,
   keysize - size of key in bits,
   ivsize - size of iv in bits
*/
void gSTREAM_keysetup(gSTREAM_ctx* ctx, u8* keys, u32 keysize, u32 ivsize);

/* Do the iv setup.
   ivs - all the stream ivs: iv[i][] corresponds to the i-th streams's iv,
*/
void gSTREAM_ivsetup(gSTREAM_ctx* ctx, u8* ivs);

/*
   inputs - all the stream inputs:
            input[i][] corresponds to the i-th streams's input,
   outputs - all the stream outputs:
            output[i][] corresponds to the i-th streams's output,
   length - input/output length in bytes
 */
void gSTREAM_process_bytes(gSTREAM_action action, gSTREAM_ctx* ctx,
                           u8* inputs, u8* outputs, u32 length);

/* Generate keystream bytes.
   keystreams[i] = keystream i
   length - keystream length in bytes
 */
void gSTREAM_keystream_bytes(gSTREAM_ctx* ctx, u8* keystreams, u32 length);

/* Free any allocated buffers and destroy context. */
void gSTREAM_exit(gSTREAM_ctx* ctx);

/* Get the measured time elapsed during keystream generation. */
double gSTREAM_getTimerValue(gSTREAM_ctx* ctx);

#endif
