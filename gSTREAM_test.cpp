#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <cutil_inline.h>
#include <cuda_runtime_api.h>
#include "gSTREAM.h"
#include "gSTREAM_test.h"

/* Allocate and generate many random keys and random ivs. */
static void gen_rand_keys_ivs(u8 **keys, size_t key_size,
                              u8 **ivs, size_t iv_size,
                              int nr) {
   unsigned i;

   if(!(*keys=(u8*)malloc(sizeof(u8)*key_size*nr))) {
      fprintf(stderr, "Failed to allocate keys: %s\n", strerror(errno));
      exit(-1);
   }

   if(!(*ivs=(u8*)malloc(sizeof(u8)*iv_size*nr))) {
      fprintf(stderr, "Failed to allocate ivs: %s\n", strerror(errno));
      exit(-1);
   }
   for(i=0;i<nr*key_size;i++) { (*keys)[i]=(u8)rand();}
   for(i=0;i<nr*iv_size;i++) { (*ivs)[i]=(u8)rand();}
}

/* Generate random buffer. */
static void gen_rand_buffs(u8 *buffs, size_t buff_size, int nr) {
   unsigned i;
   for(i=0;i<nr*buff_size;i++) {
      buffs[i]=(u8)rand();
   }
}

void do_test(int seed, int dev_no, int nr_threads, int nr_blocks,
             gSTREAM_action action,
             size_t key_size_bytes,
             size_t iv_size_bytes, size_t buff_size_bytes) {


   gSTREAM_ctx ctx;
   u8 *keys, *ivs, *buffs;
   int nr_streams=nr_threads*nr_blocks;
   double ms_time;

   srand(seed);

   gen_rand_keys_ivs(&keys,key_size_bytes,&ivs,iv_size_bytes,nr_streams);

   if(PRINT_KEYIV){
      /* print keys and ivs */
      unsigned i;
      printf("Keys:\n");
      for(i=0;i<nr_streams*key_size_bytes;i++) {
         printf("0x%02x, ",keys[i]);
         if(!((i+1)%key_size_bytes)) { printf("\n"); }
      }

      printf("IVs:\n");
      for(i=0;i<nr_streams*iv_size_bytes;i++) {
         printf("0x%02x, ",ivs[i]);
         if(!((i+1)%iv_size_bytes)) { printf("\n"); }
      }
   }

   if(!(buffs=(u8*)malloc(sizeof(u8)*buff_size_bytes*nr_streams))) {
      fprintf(stderr, "Failed to allocate buffs: %s\n", strerror(errno));
      exit(-1);
   }

   /* initialize context */
   gSTREAM_init(&ctx,dev_no,nr_threads,nr_blocks);

   /* do the key and iv setup */
   gSTREAM_keysetup(&ctx,(u8*)keys,key_size_bytes*8,iv_size_bytes*8);
   gSTREAM_ivsetup(&ctx,(u8*)ivs);

   if(action==GEN_KEYSTREAM) {
      gSTREAM_keystream_bytes(&ctx,(u8*)buffs,buff_size_bytes);
   } else {
      gen_rand_buffs(buffs,buff_size_bytes,nr_streams);
      if(PRINT_INPUT) {
         /* print input */
         unsigned i;
         printf("Input:\n");
         for(i=0;i<nr_streams*buff_size_bytes;i++) {
            printf("0x%02x, ",buffs[i]);
            if(!((i+1)%buff_size_bytes)) { printf("\n"); }
         }
      }
      gSTREAM_process_bytes(ENCRYPT,&ctx,(u8*)buffs,(u8*)buffs,buff_size_bytes);
   }



   if(PRINT_OUTPUT){
      /* print output */
      unsigned i;
      printf("Output:\n");
      for(i=0;i<nr_streams*buff_size_bytes;i++) {
         printf("0x%02x, ",buffs[i]);
         if(!((i+1)%buff_size_bytes)) { printf("\n"); }
      }
   }

   ms_time=gSTREAM_getTimerValue(&ctx);

   debug("Elapsed time: %f ms, %f cycles/byte\n",ms_time
             ,1242*1000*ms_time/(buff_size_bytes*nr_streams));

   gSTREAM_exit(&ctx);

   free(keys);
   free(ivs);
   free(buffs);

}
