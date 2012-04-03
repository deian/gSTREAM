#ifndef __Salsa20_KERNEL_CU__
#define __Salsa20_KERNEL_CU__

#define __mem(mm,i,j,N) ((mm)[(i)+(j)*(N)])
#define max(a,b) (((a)>(b))?(a):(b))
#define min(a,b) (((a)<(b))?(a):(b))

#define rotl32(v, n) \
     ((u32)((v) << (n)) | ((v) >> (32 - (n))))

#define ROTATE(v,c) (rotl32(v,c))
#define XOR(v,w) ((v) ^ (w))
#define PLUS(v,w) ((u32)((v) + (w)))
#define PLUSONE(v) (PLUS((v),1))

#define SIGMA_0 0x61707865
#define SIGMA_1 0x3320646e
#define SIGMA_2 0x79622d32
#define SIGMA_3 0x6b206574
#define TAU_0   0x61707865
#define TAU_1   0x3120646e
#define TAU_2   0x79622d36
#define TAU_3   0x6b206574



__global__ void Salsa20_keyivsetup(u32* g_x,
                                   u32 *keys, u32 key_size,
                                   u32 *ivs, u32 iv_size) {
   u32 tID=blockIdx.x*blockDim.x+threadIdx.x;
   u32 nr_streams=blockDim.x*gridDim.x;

   u32 x0, x1, x2, x3, x4, x5, x10, x11, x12, x13, x14, x15;

   x1 = __mem(keys,tID,0,nr_streams);
   x2 = __mem(keys,tID,1,nr_streams);
   x3 = __mem(keys,tID,2,nr_streams);
   x4 = __mem(keys,tID,3,nr_streams);

   if(key_size==256) {
      x11 = __mem(keys,tID,4,nr_streams);
      x12 = __mem(keys,tID,5,nr_streams);
      x13 = __mem(keys,tID,6,nr_streams);
      x14 = __mem(keys,tID,7,nr_streams);
      x0  = SIGMA_0;
      x5  = SIGMA_1;
      x10 = SIGMA_2;
      x15 = SIGMA_3;
   } else {
      x11 = x1;
      x12 = x2;
      x13 = x3;
      x14 = x4;
      x0  = TAU_0;
      x5  = TAU_1;
      x10 = TAU_2;
      x15 = TAU_3;
   }
   __mem(g_x,tID, 0,nr_streams) = x0;
   __mem(g_x,tID, 1,nr_streams) = x1;
   __mem(g_x,tID, 2,nr_streams) = x2;
   __mem(g_x,tID, 3,nr_streams) = x3;
   __mem(g_x,tID, 4,nr_streams) = x4;
   __mem(g_x,tID, 5,nr_streams) = x5;
   if(iv_size>0) {
      __mem(g_x,tID, 6,nr_streams) = __mem(ivs,tID,0,nr_streams);
      __mem(g_x,tID, 7,nr_streams) = __mem(ivs,tID,1,nr_streams);
   }
   __mem(g_x,tID, 8,nr_streams) = 0;
   __mem(g_x,tID, 9,nr_streams) = 0;
   __mem(g_x,tID,10,nr_streams) = x10;
   __mem(g_x,tID,11,nr_streams) = x11;
   __mem(g_x,tID,12,nr_streams) = x12;
   __mem(g_x,tID,13,nr_streams) = x13;
   __mem(g_x,tID,14,nr_streams) = x14;
   __mem(g_x,tID,15,nr_streams) = x15;
  
}
#define print_all\
      printf("%d:[0x%08x],[0x%08x]\n",tID,x0 ,input( 0));\
      printf("%d:[0x%08x],[0x%08x]\n",tID,x1 ,input( 1));\
      printf("%d:[0x%08x],[0x%08x]\n",tID,x2 ,input( 2));\
      printf("%d:[0x%08x],[0x%08x]\n",tID,x3 ,input( 3));\
      printf("%d:[0x%08x],[0x%08x]\n",tID,x4 ,input( 4));\
      printf("%d:[0x%08x],[0x%08x]\n",tID,x5 ,input( 5));\
      printf("%d:[0x%08x],[0x%08x]\n",tID,x6 ,input( 6));\
      printf("%d:[0x%08x],[0x%08x]\n",tID,x7 ,input( 7));\
      printf("%d:[0x%08x],[0x%08x]\n",tID,x8 ,input( 8));\
      printf("%d:[0x%08x],[0x%08x]\n",tID,x9 ,input( 9));\
      printf("%d:[0x%08x],[0x%08x]\n",tID,x10,input(10));\
      printf("%d:[0x%08x],[0x%08x]\n",tID,x11,input(11));\
      printf("%d:[0x%08x],[0x%08x]\n",tID,x12,input(12));\
      printf("%d:[0x%08x],[0x%08x]\n",tID,x13,input(13));\
      printf("%d:[0x%08x],[0x%08x]\n",tID,x14,input(14));\
      printf("%d:[0x%08x],[0x%08x]\n",tID,x15,input(15));\
      printf("\n");\


#define SALSA20(x)\
   do {\
      int i;\
      x0  = input( 0);\
      x1  = input( 1);\
      x2  = input( 2);\
      x3  = input( 3);\
      x4  = input( 4);\
      x5  = input( 5);\
      x6  = input( 6);\
      x7  = input( 7);\
      x8  = input( 8);\
      x9  = input( 9);\
      x10 = input(10);\
      x11 = input(11);\
      x12 = input(12);\
      x13 = input(13);\
      x14 = input(14);\
      x15 = input(15);\
      for (i = 20;i > 0;i -= 2) {\
         x4 = XOR( x4,ROTATE(PLUS( x0,x12), 7));\
         x8 = XOR( x8,ROTATE(PLUS( x4, x0), 9));\
         x12 = XOR(x12,ROTATE(PLUS( x8, x4),13));\
         x0 = XOR( x0,ROTATE(PLUS(x12, x8),18));\
         x9 = XOR( x9,ROTATE(PLUS( x5, x1), 7));\
         x13 = XOR(x13,ROTATE(PLUS( x9, x5), 9));\
         x1 = XOR( x1,ROTATE(PLUS(x13, x9),13));\
         x5 = XOR( x5,ROTATE(PLUS( x1,x13),18));\
         x14 = XOR(x14,ROTATE(PLUS(x10, x6), 7));\
         x2 = XOR( x2,ROTATE(PLUS(x14,x10), 9));\
         x6 = XOR( x6,ROTATE(PLUS( x2,x14),13));\
         x10 = XOR(x10,ROTATE(PLUS( x6, x2),18));\
         x3 = XOR( x3,ROTATE(PLUS(x15,x11), 7));\
         x7 = XOR( x7,ROTATE(PLUS( x3,x15), 9));\
         x11 = XOR(x11,ROTATE(PLUS( x7, x3),13));\
         x15 = XOR(x15,ROTATE(PLUS(x11, x7),18));\
         x1 = XOR( x1,ROTATE(PLUS( x0, x3), 7));\
         x2 = XOR( x2,ROTATE(PLUS( x1, x0), 9));\
         x3 = XOR( x3,ROTATE(PLUS( x2, x1),13));\
         x0 = XOR( x0,ROTATE(PLUS( x3, x2),18));\
         x6 = XOR( x6,ROTATE(PLUS( x5, x4), 7));\
         x7 = XOR( x7,ROTATE(PLUS( x6, x5), 9));\
         x4 = XOR( x4,ROTATE(PLUS( x7, x6),13));\
         x5 = XOR( x5,ROTATE(PLUS( x4, x7),18));\
         x11 = XOR(x11,ROTATE(PLUS(x10, x9), 7));\
         x8 = XOR( x8,ROTATE(PLUS(x11,x10), 9));\
         x9 = XOR( x9,ROTATE(PLUS( x8,x11),13));\
         x10 = XOR(x10,ROTATE(PLUS( x9, x8),18));\
         x12 = XOR(x12,ROTATE(PLUS(x15,x14), 7));\
         x13 = XOR(x13,ROTATE(PLUS(x12,x15), 9));\
         x14 = XOR(x14,ROTATE(PLUS(x13,x12),13));\
         x15 = XOR(x15,ROTATE(PLUS(x14,x13),18));\
      }\
      x0 = PLUS( x0,input( 0));\
      x1 = PLUS( x1,input( 1));\
      x2 = PLUS( x2,input( 2));\
      x3 = PLUS( x3,input( 3));\
      x4 = PLUS( x4,input( 4));\
      x5 = PLUS( x5,input( 5));\
      x6 = PLUS( x6,input( 6));\
      x7 = PLUS( x7,input( 7));\
      x8 = PLUS( x8,input( 8));\
      x9 = PLUS( x9,input( 9));\
      x10 = PLUS(x10,input(10));\
      x11 = PLUS(x11,input(11));\
      x12 = PLUS(x12,input(12));\
      x13 = PLUS(x13,input(13));\
      x14 = PLUS(x14,input(14));\
      x15 = PLUS(x15,input(15));\
      if(!(input( 8) = PLUSONE(input( 8)))) {\
         input( 9) = PLUSONE(input( 9));\
      }\
   } while(0)


extern __shared__ __align__ (__alignof(void*)) u32 smem_cache[];

__global__ void Salsa20_process_blocks(gSTREAM_action act, u32* g_x,
                                       u32 *buff, u32 nr_blocks) {
   u32 tID=blockIdx.x*blockDim.x+threadIdx.x;
   u32 nr_streams=blockDim.x*gridDim.x;

   u32 x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;
#ifdef INPUT_SHMEM
   u32* input_csh=(u32*) smem_cache;
#define input(idx) __mem(input_csh,threadIdx.x,(idx),blockDim.x)
#else
   u32 i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15;
#define input(idx) i##idx
#endif


   /* load state */
   input( 0) = __mem(g_x,tID, 0,nr_streams);
   input( 1) = __mem(g_x,tID, 1,nr_streams);
   input( 2) = __mem(g_x,tID, 2,nr_streams);
   input( 3) = __mem(g_x,tID, 3,nr_streams);
   input( 4) = __mem(g_x,tID, 4,nr_streams);
   input( 5) = __mem(g_x,tID, 5,nr_streams);
   input( 6) = __mem(g_x,tID, 6,nr_streams);
   input( 7) = __mem(g_x,tID, 7,nr_streams);
   input( 8) = __mem(g_x,tID, 8,nr_streams);
   input( 9) = __mem(g_x,tID, 9,nr_streams);
   input(10) = __mem(g_x,tID,10,nr_streams);
   input(11) = __mem(g_x,tID,11,nr_streams);
   input(12) = __mem(g_x,tID,12,nr_streams);
   input(13) = __mem(g_x,tID,13,nr_streams);
   input(14) = __mem(g_x,tID,14,nr_streams);
   input(15) = __mem(g_x,tID,15,nr_streams);



   for(int block_no=0;block_no<nr_blocks;block_no++) {
      
      /* output of Salsa20 is x0 - x 15 */
      SALSA20(x);

      /* copy/xor-into global buffer */
      if(act!=GEN_KEYSTREAM) {
         __mem(buff,tID, 0,nr_streams) ^= x0;
         __mem(buff,tID, 1,nr_streams) ^= x1;
         __mem(buff,tID, 2,nr_streams) ^= x2;
         __mem(buff,tID, 3,nr_streams) ^= x3;
         __mem(buff,tID, 4,nr_streams) ^= x4;
         __mem(buff,tID, 5,nr_streams) ^= x5;
         __mem(buff,tID, 6,nr_streams) ^= x6;
         __mem(buff,tID, 7,nr_streams) ^= x7;
         __mem(buff,tID, 8,nr_streams) ^= x8;
         __mem(buff,tID, 9,nr_streams) ^= x9;
         __mem(buff,tID,10,nr_streams) ^= x10;
         __mem(buff,tID,11,nr_streams) ^= x11;
         __mem(buff,tID,12,nr_streams) ^= x12;
         __mem(buff,tID,13,nr_streams) ^= x13;
         __mem(buff,tID,14,nr_streams) ^= x14;
         __mem(buff,tID,15,nr_streams) ^= x15;
      } else {
         __mem(buff,tID, 0,nr_streams)  = x0;
         __mem(buff,tID, 1,nr_streams)  = x1;
         __mem(buff,tID, 2,nr_streams)  = x2;
         __mem(buff,tID, 3,nr_streams)  = x3;
         __mem(buff,tID, 4,nr_streams)  = x4;
         __mem(buff,tID, 5,nr_streams)  = x5;
         __mem(buff,tID, 6,nr_streams)  = x6;
         __mem(buff,tID, 7,nr_streams)  = x7;
         __mem(buff,tID, 8,nr_streams)  = x8;
         __mem(buff,tID, 9,nr_streams)  = x9;
         __mem(buff,tID,10,nr_streams)  = x10;
         __mem(buff,tID,11,nr_streams)  = x11;
         __mem(buff,tID,12,nr_streams)  = x12;
         __mem(buff,tID,13,nr_streams)  = x13;
         __mem(buff,tID,14,nr_streams)  = x14;
         __mem(buff,tID,15,nr_streams)  = x15;
      }
      buff+=16*nr_streams;

   }
   /* save state */
   __mem(g_x,tID, 8,nr_streams) = input( 8);
   __mem(g_x,tID, 9,nr_streams) = input( 9);

}
#endif


