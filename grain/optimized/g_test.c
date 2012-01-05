#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "ecrypt-sync.h"

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

void do_test(int seed,
      int action, /* -1 - keygen, 0 - encrypt, 1 - decrypt*/
      int nr_threads, int nr_blocks,
      size_t key_size_bytes, size_t iv_size_bytes, size_t buff_size_bytes)
{
   srand(seed);
   ECRYPT_ctx ctx;
   ECRYPT_init ();
   int nr_streams=nr_threads*nr_blocks;
   unsigned i,stream;

   u8 *buffs, *buff;
   u8 *keys, *ivs;
   u8 *key, *iv;

   gen_rand_keys_ivs(&keys,key_size_bytes,&ivs,iv_size_bytes,nr_streams);

   if(!(buffs=(u8*)malloc(sizeof(u8)*buff_size_bytes*nr_streams))) {
      fprintf(stderr, "Failed to allocate buffs: %s\n", strerror(errno));
      exit(-1);
   }

   if(action!=-1) {
      gen_rand_buffs(buffs,buff_size_bytes,nr_streams);
   }

   printf("Output:\n");
   buff=buffs; key=keys; iv=ivs;
   for(stream=0;stream<nr_streams;stream++) {

      ECRYPT_keysetup(&ctx,key,key_size_bytes*8,iv_size_bytes*8);
      ECRYPT_ivsetup(&ctx,iv);

      if(action!=-1) {
         ECRYPT_process_bytes(action,&ctx,buff,buff,buff_size_bytes);
      } else {
         ECRYPT_keystream_bytes(&ctx,buff,buff_size_bytes);
      }

      for(i=0;i<buff_size_bytes;i++) {
         printf("0x%02x, ",buff[i]);
      }
      printf("\n");

      buff+=buff_size_bytes;
      key+=key_size_bytes;
      iv+=iv_size_bytes;

   }

   free(keys);
   free(ivs);
   free(buffs);

}

int main() {
   do_test(0,-1
          ,128,680
          ,10,8,2048);

   /*
   int i;
   u8 key[]=
   {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
//   {0xC2, 0x1F, 0xCF, 0x38, 0x81, 0xCD, 0x5E, 0xE8, 0x62, 0x8A, 0xCC, 0xB0, 0xA9, 0x89, 0x0D, 0xF8};

   u8 iv[]=
//   {0x27, 0x17, 0xF4, 0xD2, 0x1A, 0x56, 0xEB, 0xA6};
//   {0x59, 0x7E, 0x26, 0xC1, 0x75, 0xF5, 0x73, 0xC3};
   {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

   u8 buff[48];
   memset(buff,0,sizeof(buff));
   ECRYPT_ctx ctx;
   ECRYPT_init ();
   ECRYPT_keysetup(&ctx,key,sizeof(key)*8,sizeof(iv)*8);
   ECRYPT_ivsetup(&ctx,iv);
   ECRYPT_keystream_bytes(&ctx,buff,sizeof(buff));
   for(i=0;i<sizeof(buff);i++) {
      printf("%02X ",buff[i]);
      if(!((i+1)%16)) {printf("\n"); }
   }
   printf("\n");
   */

   return 0;
}

