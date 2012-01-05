#ifndef __Grain_KERNEL_CU__
#define __Grain_KERNEL_CU__

#define __mem(mm,i,j,N) ((mm)[(i)+(j)*(N)])
#define max(a,b) (((a)>(b))?(a):(b))
#define min(a,b) (((a)<(b))?(a):(b))

#define INITCLOCKS 160

#define S0 (s0)
#define S1 (s1)
#define S2 (s2)

#define B0 (b0)
#define B1 (b1)
#define B2 (b2)

#define get64(S) (S##2)
#define get63(S) (S##1>>31)
#define get62(S) (S##1>>30)
#define get60(S) (S##1>>28)
#define get56(S) (S##1>>24)
#define get52(S) (S##1>>20)
#define get51(S) (S##1>>19)
#define get46(S) (S##1>>14)
#define get45(S) (S##1>>13)
#define get43(S) (S##1>>11)
#define get38(S) (S##1>> 6)
#define get37(S) (S##1>> 5)
#define get33(S) (S##1>> 1)
#define get31(S) (S##0>>31)
#define get28(S) (S##0>>28)
#define get25(S) (S##0>>25)
#define get23(S) (S##0>>23)
#define get21(S) (S##0>>21)
#define get15(S) (S##0>>15)
#define get14(S) (S##0>>14)
#define get13(S) (S##0>>13)
#define get10(S) (S##0>>10)
#define get9(S)  (S##0>> 9)
#define get4(S)  (S##0>> 4)
#define get3(S)  (S##0>> 3)
#define get2(S)  (S##0>> 2)
#define get1(S)  (S##0>> 1)
#define get0(S)  (S##0)

#define set79(S,bit) (S##2=(S##2&(~(1<<15)))|((bit&1)<<15))
#define xor79(S,bit) (S##2^=((bit&1)<<15))

#define h(x0,x1,x2,x3,x4,x02,x24)\
   ((x1)^(x4)^((x0)&(x3))^((x2)&(x3))^((x3)&(x4))^((x02)&(x1))^((x02)&(x3))^((x02)&(x4))^((x1)&(x24))^((x24)&(x3)))

#define SHIFT_FSR(S)\
      do {\
         S##0=(S##0>>1)|(((S##1)&1)<<31);\
         S##1=(S##1>>1)|(((S##2)&1)<<31);\
         S##2=(S##2>>1);\
      } while(0)

#define Grain_keystream(Z)\
   do {\
      u32 x0  = get3(S),\
          x1  = get25(S),\
          x2  = get46(S),\
          x3  = get64(S),\
          x4  = get63(B),\
          x02 = x0&x2,\
          x24 = x2&x4;\
\
      Z   = (get1(B) ^ get2(B) ^ get4(B) ^ get10(B) ^ get31(B) ^ get43(B) ^ get56(B) ^ h(x0,x1,x2,x3,x4,x02,x24))&1;\
      u32 S80 = get62(S) ^ get51(S) ^ get38(S) ^ get23(S) ^ get13(S) ^ get0(S);\
\
      u32 B33_28_21 = (get33(B)&get28(B)&get21(B));\
      u32 B52_45_37 = (get52(B)&get45(B)&get37(B));\
      u32 B52_37_33 = (get52(B)&get37(B)&get33(B));\
      u32 B60_52_45 = (get60(B)&get52(B)&get45(B));\
      u32 B63_60 = (get63(B)&get60(B));\
      u32 B37_33 = (get37(B)&get33(B));\
      u32 B45_28 = (get45(B)&get28(B));\
      u32 B15_9  = (get15(B)&get9(B)); \
      u32 B21_15 = (get21(B)&get15(B));\
\
      u32 B80 =(get0(S)) ^ (get62(B)) ^ (get60(B)) ^ (get52(B)) ^ (get45(B)) ^ (get37(B)) ^ (get33(B)) ^ (get28(B)) ^ (get21(B))^\
         (get14(B)) ^ (get9(B)) ^ (get0(B)) ^ (B63_60) ^ (B37_33) ^ (B15_9)^\
         (B60_52_45) ^ (B33_28_21) ^ (get63(B)&B45_28&get9(B))^\
         (get60(B)&B52_37_33) ^ (B63_60&B21_15)^\
         (B63_60&B52_45_37) ^ (B33_28_21&B15_9)^\
         (B52_45_37&B33_28_21);\
\
      SHIFT_FSR(S);\
      SHIFT_FSR(B);\
      set79(S,S80);\
      set79(B,B80);\
   } while(0)


__global__ void Grain_keyivsetup(u32* g_s, u32 *g_b,
                                 u32 *keys, u32 key_size,
                                 u32 *ivs, u32 iv_size) {
   u32 tID=blockIdx.x*blockDim.x+threadIdx.x;
   u32 nr_streams=blockDim.x*gridDim.x;

   u32 k0, k1, k2;
   u32 i0, i1;

   u32 s0,s1,s2;                   u32 b0,b1,b2;

   k0=__mem(keys,tID,0,nr_streams); i0=__mem(ivs,tID,0,nr_streams);
   k1=__mem(keys,tID,1,nr_streams); i1=__mem(ivs,tID,1,nr_streams);
   k2=__mem(keys,tID,2,nr_streams);

   /* load key */
   b0=k0;
   b1=k1;
   b2=k2&0xffff;

   /* load iv */
   s0=i0;
   s1=i1;
   s2=0xffff;

   u32 Z;
   for(int i=0;i<INITCLOCKS;++i) {
      Grain_keystream(Z);
      xor79(S,Z);
      xor79(B,Z);
   }

   __mem(g_s,tID,0,nr_streams)=s0; __mem(g_b,tID,0,nr_streams)=b0;
   __mem(g_s,tID,1,nr_streams)=s1; __mem(g_b,tID,1,nr_streams)=b1;
   __mem(g_s,tID,2,nr_streams)=s2; __mem(g_b,tID,2,nr_streams)=b2;

  
}


__global__ void Grain_process_bytes(gSTREAM_action act, u32* g_s, u32 *g_b,
                                    u32 *buff, u32 nr_words) {
   u32 tID=blockIdx.x*blockDim.x+threadIdx.x;
   u32 nr_streams=blockDim.x*gridDim.x;

   u32 s0,s1,s2;                   u32 b0,b1,b2;
   s0=__mem(g_s,tID,0,nr_streams); b0=__mem(g_b,tID,0,nr_streams);
   s1=__mem(g_s,tID,1,nr_streams); b1=__mem(g_b,tID,1,nr_streams);
   s2=__mem(g_s,tID,2,nr_streams); b2=__mem(g_b,tID,2,nr_streams);

   for(int w=0;w<nr_words;w++) {
      u32 output_word=0;

      if(act!=GEN_KEYSTREAM) {
         output_word=__mem(buff,tID,w,nr_streams);
      }
#pragma unroll 322
      for(int i=0;i<32;i++) {
         u32 Z;
         Grain_keystream(Z);
         output_word ^= Z << i;
      }
      __mem(buff,tID,w,nr_streams)=output_word;
   }

   __mem(g_s,tID,0,nr_streams)=s0; __mem(g_b,tID,0,nr_streams)=b0;
   __mem(g_s,tID,1,nr_streams)=s1; __mem(g_b,tID,1,nr_streams)=b1;
   __mem(g_s,tID,2,nr_streams)=s2; __mem(g_b,tID,2,nr_streams)=b2;

}
#endif


