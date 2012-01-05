#include <cutil_inline.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include "gSTREAM.h"

/* include cipher kernel function cu file */
#include "Grain_kernel.cu"


void gSTREAM_init(gSTREAM_ctx* ctx, int device, int nr_threads, int nr_blocks){

   cudaDeviceProp deviceProp;
   int nr_streams=nr_threads*nr_blocks;

   /* set device */
   cudaGetDeviceProperties(&deviceProp, device);
   cudaSetDevice(device);
   debug("\nUsing device %d: \"%s\"\n", device, deviceProp.name);

   cutilSafeCall(cudaSetDeviceFlags(cudaDeviceMapHost));

   ctx->nr_threads = nr_threads;
   ctx->nr_blocks = nr_blocks;
   ctx->allocated_keys=0;
   ctx->allocated_ivs=0;
   ctx->allocated_buff=0;

   cutilCheckError(cutCreateTimer(&(ctx->bench.timer)));

   /* allocate cipher state */
   Grain_ctx *gctx=&ctx->gctx;
   cutilSafeCall(cudaMalloc((void**)&(gctx->s_d),nr_streams*3*sizeof(u32)));
   cutilSafeCall(cudaMalloc((void**)&(gctx->b_d),nr_streams*3*sizeof(u32)));

}

void gSTREAM_exit(gSTREAM_ctx* ctx) {

   if(ctx->allocated_keys) {
      cutilSafeCall(cudaFree(ctx->keys_d));
   }

   if(ctx->allocated_ivs) {
      cutilSafeCall(cudaFree(ctx->ivs_d));
   }

   if(ctx->allocated_buff) {
      cutilSafeCall(cudaFreeHost(ctx->buff_h));
   }

   cutilCheckError(cutDeleteTimer(ctx->bench.timer));

   /* free cipher state */
   Grain_ctx *gctx=&ctx->gctx;
   cutilSafeCall(cudaFree(gctx->s_d));
}

void gSTREAM_keysetup(gSTREAM_ctx* ctx, u8* keys, u32 keysize, u32 ivsize) {

   size_t keys_size;
   int nr_streams=ctx->nr_threads*ctx->nr_blocks;
   u32* keys_h=NULL;
   size_t key_size_bytes=sizeof(u8)*(((keysize-1)/(sizeof(u8)*8))+1);
   size_t key_size_nrwords=(((keysize-1)/(sizeof(u32)*8))+1);

   ctx->key_size=keysize;
   ctx->iv_size=ivsize;

   /* allocate keys */
   keys_size=nr_streams*sizeof(u32)*(((keysize-1)/(sizeof(u32)*8))+1);
   cutilSafeCall(cudaMalloc((void**)&(ctx->keys_d),keys_size));
   ctx->allocated_keys=1;
   if(!(keys_h=(u32*)malloc(keys_size))) {
      fprintf(stderr,"Could not allocate keys_h: %s\n",strerror(errno));
      exit(-1);
   }

   /* copy byte-aligned keys to word-stream-aligned keys */
   {
      u32  *curr_key;
      u8* tmp_keys=keys;

      /* allocate a current working key */
      if(!(curr_key=(u32*)malloc(sizeof(u32)*key_size_nrwords))) {
         fprintf(stderr,"Could not allocate curr_key: %s\n",strerror(errno));
         exit(-1);
      }
      memset(curr_key,0x00,sizeof(u32)*key_size_nrwords);

      for(int i=0;i<nr_streams;i++) {
         /* copy one of the keys to current key */
         memcpy(curr_key,tmp_keys,key_size_bytes);
         tmp_keys+=key_size_bytes;
         /* copy current key to stream-aligned one */
         for(int j=0;j<key_size_nrwords;j++) {
            keys_h[j*nr_streams+i]=CH_ENDIANESS32(curr_key[j]);
         }
      }

      free(curr_key);
   }


   /* Copy keys to device and free them from host */
   cutilSafeCall(cudaMemcpy(ctx->keys_d,keys_h,keys_size,
                                          cudaMemcpyHostToDevice));
   free(keys_h);

}

void gSTREAM_ivsetup(gSTREAM_ctx* ctx, u8* ivs) {

   int nr_streams=ctx->nr_threads*ctx->nr_blocks;
   /* initialize the registers to all zeros */

   if(ctx->iv_size>0) {
      u8* tmp_ivs=ivs;
      u32* ivs_h=NULL;
      size_t ivs_size=
         nr_streams*sizeof(u32)*(((ctx->iv_size-1)/(sizeof(u32)*8))+1);

      u32  *curr_iv;
      size_t iv_size_bytes=sizeof(u8)*(((ctx->iv_size-1)/(sizeof(u8)*8))+1);
      size_t iv_size_nrwords=(((ctx->iv_size-1)/(sizeof(u32)*8))+1);

      cutilSafeCall(cudaMalloc((void**)&(ctx->ivs_d),ivs_size));
      ctx->allocated_ivs=1;

      if(!(ivs_h=(u32*)malloc(ivs_size))) {
         fprintf(stderr,"Could not allocate ivs_h: %s\n",strerror(errno));
         exit(-1);
      }

      /* allocate a current working iv */
      if(!(curr_iv=(u32*)malloc(sizeof(u32)*iv_size_nrwords))) {
         fprintf(stderr,"Could not allocate curr_iv: %s\n",strerror(errno));
         exit(-1);
      }
      memset(curr_iv,0x00,sizeof(u32)*iv_size_nrwords);

      for(int i=0;i<nr_streams;i++) {
         /* copy one of the ivs to current iv */
         memcpy(curr_iv,tmp_ivs,iv_size_bytes);
         tmp_ivs+=iv_size_bytes;
         /* copy current iv to stream-aligned one */
         for(int j=0;j<iv_size_nrwords;j++) {
            ivs_h[j*nr_streams+i]=CH_ENDIANESS32(curr_iv[j]);
         }
      }
      free(curr_iv);

      /* Copy ivs to device and free them from host */
      cutilSafeCall(cudaMemcpy(ctx->ivs_d,ivs_h,ivs_size,
                                                cudaMemcpyHostToDevice));
      free(ivs_h);
   }

   /* Load in iv, key and preclock */
   Grain_ctx *gctx=&ctx->gctx;
   Grain_keyivsetup<<<ctx->nr_blocks,ctx->nr_threads>>>(gctx->s_d
                                                       ,gctx->b_d
                                                       ,ctx->keys_d
                                                       ,ctx->key_size
                                                       ,ctx->ivs_d
                                                       ,ctx->iv_size);
   cutilCheckMsg("Kernel execution failed");
   cudaThreadSynchronize();
#if 0
   {//print state, each colum corresponds to a different stream
      u32 *s_h, *b_h;
      if(!(s_h=(u32*)malloc(nr_streams*3*sizeof(u32)))) {
         fprintf(stderr, "Failed to allocate s_h: %s\n",strerror(errno));
         exit(-1);
      }
      if(!(b_h=(u32*)malloc(nr_streams*3*sizeof(u32)))) {
         fprintf(stderr, "Failed to allocate g_h: %s\n",strerror(errno));
         exit(-1);
      }

      cutilSafeCall(cudaMemcpy(s_h,gctx->s_d,(nr_streams*3*sizeof(u32)),
               cudaMemcpyDeviceToHost));
      cutilSafeCall(cudaMemcpy(b_h,gctx->b_d,(nr_streams*3*sizeof(u32)),
               cudaMemcpyDeviceToHost));

      for(int i=0;i<nr_streams*3;i++) {
         printf("[0x%08x:0x%08x], ",s_h[i],b_h[i]);
         if(!((i+1)%nr_streams)) { printf("\n"); }
      }

      free(s_h);
      free(b_h);
   }
#endif

}

void gSTREAM_keystream_bytes(gSTREAM_ctx* ctx, u8* keystreams, u32 length) {
   gSTREAM_process_bytes(GEN_KEYSTREAM,ctx,NULL,keystreams,length);
}

void gSTREAM_process_bytes(gSTREAM_action action, gSTREAM_ctx* ctx,
                                       u8* inputs, u8* outputs, u32 length) {
   int nr_streams=ctx->nr_blocks*ctx->nr_threads;
   size_t length_nr_words=(((length-1)/(sizeof(u32)))+1);
   size_t buff_size=nr_streams*length_nr_words*sizeof(u32);
   u32* tmp_buffer;

   /* allocate buffer */
   if((!ctx->allocated_buff)||((length_nr_words*sizeof(u32))>ctx->buff_size)) {
      if(ctx->allocated_buff) {
         free(ctx->buff_h); //alocate a large buffer
      }
      cutilSafeCall(cudaHostAlloc((void**)&(ctx->buff_h),buff_size,
               cudaHostAllocMapped));
      cutilSafeCall(cudaHostGetDevicePointer((void **)&(ctx->buff_d),
               ctx->buff_h,0));
      ctx->allocated_buff=1;
      ctx->buff_size=length_nr_words*sizeof(u32);
   }

   /* allocate a current working buffer */
   if(!(tmp_buffer=(u32*)malloc(sizeof(u32)*length_nr_words))) {
      fprintf(stderr,"Could not allocate tmp_buffer: %s\n",strerror(errno));
      exit(-1);
   }

   if(action!=GEN_KEYSTREAM) {
      for(int i=0;i<nr_streams;i++) {
         /* copy one of the inputs to current working buffer */
         memcpy(tmp_buffer,inputs,length);
         inputs+=length;
         /* copy current iv to stream-aligned one */
         for(int j=0;j<length_nr_words;j++) {
            ctx->buff_h[j*nr_streams+i]=CH_ENDIANESS32(tmp_buffer[j]);
         }
      }
   }

   /* process bytes */
   Grain_ctx *gctx=&ctx->gctx;
   cutilCheckError(cutStartTimer(ctx->bench.timer));
   Grain_process_bytes<<<ctx->nr_blocks,ctx->nr_threads>>>(action
                                                          ,gctx->s_d
                                                          ,gctx->b_d
                                                          ,ctx->buff_d
                                                          ,length_nr_words);
   cutilCheckMsg("Kernel execution failed");
   cudaThreadSynchronize();
   cutilCheckError(cutStopTimer(ctx->bench.timer));

   /* copy from working buffer to output buffer */
   for(int i=0;i<nr_streams;i++) {
      /* copy one of the keystreams to current keystream */
      for(int j=0;j<length_nr_words;j++) {
         tmp_buffer[j]=ctx->buff_h[i+j*nr_streams];
      }
      memcpy(outputs,tmp_buffer,length);
      outputs+=length;
   }

   free(tmp_buffer);
}

double gSTREAM_getTimerValue(gSTREAM_ctx* ctx) {
   return cutGetTimerValue(ctx->bench.timer);
}
