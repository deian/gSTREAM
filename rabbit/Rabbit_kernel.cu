#ifndef __Rabbit_KERNEL_CU__
#define __Rabbit_KERNEL_CU__

#define __mem(mm,i,j,N) ((mm)[(i)+(j)*(N)])
#define max(a,b) (((a)>(b))?(a):(b))
#define min(a,b) (((a)<(b))?(a):(b))

#define rotl32(v, n) \
     ((u32)((v) << (n)) | ((v) >> (32 - (n))))


#define DEFINE_STATE(x,c,carry)\
   u32 x##0, x##1, x##2, x##3,\
       x##4, x##5, x##6, x##7;\
   u32 c##0, c##1, c##2, c##3,\
       c##4, c##5, c##6, c##7;\
   u32 carry

#define LOAD8(s,g_s)\
   do {\
         s##0 = __mem((g_s),tID,0,nr_streams);\
         s##1 = __mem((g_s),tID,1,nr_streams);\
         s##2 = __mem((g_s),tID,2,nr_streams);\
         s##3 = __mem((g_s),tID,3,nr_streams);\
         s##4 = __mem((g_s),tID,4,nr_streams);\
         s##5 = __mem((g_s),tID,5,nr_streams);\
         s##6 = __mem((g_s),tID,6,nr_streams);\
         s##7 = __mem((g_s),tID,7,nr_streams);\
   } while(0)

#define LOAD_STATE(x,g_x,c,g_c,carry,g_carry)\
   do {\
      LOAD8(x,g_x); LOAD8(c,g_c);\
      carry = __mem((g_carry),tID,0,nr_streams);\
   } while(0)

#define STORE8(s,g_s)\
   do {\
         __mem((g_s),tID,0,nr_streams) = s##0;\
         __mem((g_s),tID,1,nr_streams) = s##1;\
         __mem((g_s),tID,2,nr_streams) = s##2;\
         __mem((g_s),tID,3,nr_streams) = s##3;\
         __mem((g_s),tID,4,nr_streams) = s##4;\
         __mem((g_s),tID,5,nr_streams) = s##5;\
         __mem((g_s),tID,6,nr_streams) = s##6;\
         __mem((g_s),tID,7,nr_streams) = s##7;\
   } while(0)

#define SAVE_STATE(x,g_x,c,g_c,carry,g_carry)\
   do {\
      STORE8(x,g_x); STORE8(c,g_c);\
      __mem((g_carry),tID,0,nr_streams) = carry;\
   } while(0)

__device__ u32 g_func(u32 x) {
   u32 a, b, h, l;
   a = x&0xFFFF;
   b = x>>16;

   h = ((((u32)(a*a)>>17) + (u32)(a*b))>>15) + b*b;
   l = x*x;

   return (u32)(h^l);
}

#define NEXT_STATE()\
   do {\
      u32 g0,g1,g2,g3,\
          g4,g5,g6,g7;\
\
      /* Temporary variables */\
      u32 c_prev,c_tmp;\
\
      /* Calculate new counter values */\
      c_prev=c0;c0 = (u32)(c0 + 0x4D34D34D + carry);\
      c_tmp=c1; c1 = (u32)(c1 + 0xD34D34D3 + (c0 < c_prev)); c_prev=c_tmp;\
      c_tmp=c2; c2 = (u32)(c2 + 0x34D34D34 + (c1 < c_prev)); c_prev=c_tmp;\
      c_tmp=c3; c3 = (u32)(c3 + 0x4D34D34D + (c2 < c_prev)); c_prev=c_tmp;\
      c_tmp=c4; c4 = (u32)(c4 + 0xD34D34D3 + (c3 < c_prev)); c_prev=c_tmp;\
      c_tmp=c5; c5 = (u32)(c5 + 0x34D34D34 + (c4 < c_prev)); c_prev=c_tmp;\
      c_tmp=c6; c6 = (u32)(c6 + 0x4D34D34D + (c5 < c_prev)); c_prev=c_tmp;\
      c_tmp=c7; c7 = (u32)(c7 + 0xD34D34D3 + (c6 < c_prev));\
      carry = (c7 < c_tmp);\
      g0=g_func((u32)(x0+c0));\
      g1=g_func((u32)(x1+c1));\
      g2=g_func((u32)(x2+c2));\
      g3=g_func((u32)(x3+c3));\
      g4=g_func((u32)(x4+c4));\
      g5=g_func((u32)(x5+c5));\
      g6=g_func((u32)(x6+c6));\
      g7=g_func((u32)(x7+c7));\
      x0 = g0;\
      x1 = g1;\
      x2 = g2;\
      x3 = g3;\
      x4 = g4;\
      x5 = g5;\
      x6 = g6;\
      x7 = g7;\
      x0 = (u32)(g0 + rotl32(g7,16) + rotl32(g6, 16));\
      x1 = (u32)(g1 + rotl32(g0, 8) + g7);\
      x2 = (u32)(g2 + rotl32(g1,16) + rotl32(g0, 16));\
      x3 = (u32)(g3 + rotl32(g2, 8) + g1);\
      x4 = (u32)(g4 + rotl32(g3,16) + rotl32(g2, 16));\
      x5 = (u32)(g5 + rotl32(g4, 8) + g3);\
      x6 = (u32)(g6 + rotl32(g5,16) + rotl32(g4, 16));\
      x7 = (u32)(g7 + rotl32(g6, 8) + g5);\
   } while(0)


__global__ void Rabbit_keysetup(u32* g_x, u32 *g_c, u32 *g_carry,
                                u32 *keys, u32 key_size) {
   u32 tID=blockIdx.x*blockDim.x+threadIdx.x;
   u32 nr_streams=blockDim.x*gridDim.x;

   u32 k0, k1, k2, k3;

   DEFINE_STATE(x,c,carry);

   k0=__mem(keys,tID,0,nr_streams);
   k1=__mem(keys,tID,1,nr_streams);
   k2=__mem(keys,tID,2,nr_streams);
   k3=__mem(keys,tID,3,nr_streams);

   x0 = k0;
   x2 = k1;
   x4 = k2;
   x6 = k3;
   x1 = (u32)(k3<<16) | (k2>>16);
   x3 = (u32)(k0<<16) | (k3>>16);
   x5 = (u32)(k1<<16) | (k0>>16);
   x7 = (u32)(k2<<16) | (k1>>16);

   c0 = rotl32(k2, 16);
   c2 = rotl32(k3, 16);
   c4 = rotl32(k0, 16);
   c6 = rotl32(k1, 16);
   c1 = (k0&0xFFFF0000) | (k1&0xFFFF);
   c3 = (k1&0xFFFF0000) | (k2&0xFFFF);
   c5 = (k2&0xFFFF0000) | (k3&0xFFFF);
   c7 = (k3&0xFFFF0000) | (k0&0xFFFF);

   carry = 0;

   for(int i=0;i<4;i++) {
      NEXT_STATE();
   }

   c0 ^= x4;
   c1 ^= x5;
   c2 ^= x6;
   c3 ^= x7;
   c4 ^= x0;
   c5 ^= x1;
   c6 ^= x2;
   c7 ^= x3;

   SAVE_STATE(x,g_x,c,g_c,carry,g_carry);

  
}

__global__ void Rabbit_ivsetup(u32* g_x, u32 *g_c, u32 *g_carry,
                                u32 *ivs, u32 iv_size) {
   u32 tID=blockIdx.x*blockDim.x+threadIdx.x;
   u32 nr_streams=blockDim.x*gridDim.x;

   u32 i0, i1, i2, i3;

   DEFINE_STATE(x,c,carry);
   LOAD_STATE(x,g_x,c,g_c,carry,g_carry);

   i0=__mem(ivs,tID,0,nr_streams);
   i2=__mem(ivs,tID,1,nr_streams);
   i1 = (i0>>16) | (i2&0xFFFF0000);
   i3 = (i2<<16) | (i0&0x0000FFFF);

   c0 ^= i0;
   c1 ^= i1;
   c2 ^= i2;
   c3 ^= i3;
   c4 ^= i0;
   c5 ^= i1;
   c6 ^= i2;
   c7 ^= i3;

   for(int i=0;i<4;i++) {
      NEXT_STATE();
   }

  
   SAVE_STATE(x,g_x,c,g_c,carry,g_carry);
}

__global__ void Rabbit_process_bytes(gSTREAM_action act, u32* g_x, u32 *g_c,
                                     u32 *g_carry, u32 *buff, u32 nr_words) {
   u32 tID=blockIdx.x*blockDim.x+threadIdx.x;
   u32 nr_streams=blockDim.x*gridDim.x;

   DEFINE_STATE(x,c,carry);
   LOAD_STATE(x,g_x,c,g_c,carry,g_carry);

   for(int w=0;w<nr_words/4;w++) {

      NEXT_STATE();

      if(act!=GEN_KEYSTREAM) {
         __mem(buff,tID,(4*w),nr_streams)   ^= x0^(x5>>16)^(u32)(x3<<16);
         __mem(buff,tID,(4*w+1),nr_streams) ^= x2^(x7>>16)^(u32)(x5<<16);
         __mem(buff,tID,(4*w+2),nr_streams) ^= x4^(x1>>16)^(u32)(x7<<16);
         __mem(buff,tID,(4*w+3),nr_streams) ^= x6^(x3>>16)^(u32)(x1<<16);
      } else {
         __mem(buff,tID,(4*w),nr_streams)    = x0^(x5>>16)^(u32)(x3<<16);
         __mem(buff,tID,(4*w+1),nr_streams)  = x2^(x7>>16)^(u32)(x5<<16);
         __mem(buff,tID,(4*w+2),nr_streams)  = x4^(x1>>16)^(u32)(x7<<16);
         __mem(buff,tID,(4*w+3),nr_streams)  = x6^(x3>>16)^(u32)(x1<<16);
      }
   }

   if(nr_words%4) {
      /* handle remaining partial 4-byte blocks */
      NEXT_STATE();
      if(act!=GEN_KEYSTREAM) {
         if((nr_words%4)==3) {
            __mem(buff,tID,(nr_words-3),nr_streams) ^= x0^(x5>>16)^(u32)(x3<<16);
            __mem(buff,tID,(nr_words-2),nr_streams) ^= x2^(x7>>16)^(u32)(x5<<16);
            __mem(buff,tID,(nr_words-1),nr_streams) ^= x4^(x1>>16)^(u32)(x7<<16);
         }                                           
         else if((nr_words%4)==2) {                  
            __mem(buff,tID,(nr_words-2),nr_streams) ^= x0^(x5>>16)^(u32)(x3<<16);
            __mem(buff,tID,(nr_words-1),nr_streams) ^= x2^(x7>>16)^(u32)(x5<<16);
         } else { //==1                               
            __mem(buff,tID,(nr_words-1),nr_streams) ^= x0^(x5>>16)^(u32)(x3<<16);
         }                                         
      } else {
         if((nr_words%4)==3) {
            __mem(buff,tID,(nr_words-3),nr_streams)  = x0^(x5>>16)^(u32)(x3<<16);
            __mem(buff,tID,(nr_words-2),nr_streams)  = x2^(x7>>16)^(u32)(x5<<16);
            __mem(buff,tID,(nr_words-1),nr_streams)  = x4^(x1>>16)^(u32)(x7<<16);
         }                                           
         else if((nr_words%4)==2) {                  
            __mem(buff,tID,(nr_words-2),nr_streams)  = x0^(x5>>16)^(u32)(x3<<16);
            __mem(buff,tID,(nr_words-1),nr_streams)  = x2^(x7>>16)^(u32)(x5<<16);
         } else { //==1                               
            __mem(buff,tID,(nr_words-1),nr_streams)  = x0^(x5>>16)^(u32)(x3<<16);
         }                                         
      }
   }

   SAVE_STATE(x,g_x,c,g_c,carry,g_carry);

}
#endif


