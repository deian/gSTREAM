#ifndef __Trivium_KERNEL_CU__
#define __Trivium_KERNEL_CU__

#define __mem(mm,i,j,N) ((mm)[(i)+(j)*(N)])
#define max(a,b) (((a)>(b))?(a):(b))
#define min(a,b) (((a)<(b))?(a):(b))

#define S(a, n) (s##a##n) 
#define T(a) (t##a)

#define S32(a, b) ((S(a, 2) << ( 64 - (b))) | (S(a, 1) >> ((b) - 32)))
#define S64(a, b) ((S(a, 3) << ( 96 - (b))) | (S(a, 2) >> ((b) - 64)))
#define S96(a, b) ((S(a, 4) << (128 - (b))) | (S(a, 3) >> ((b) - 96)))

#define UPDATE()                                                             \
  do {                                                                       \
    T(1) = S64(1,  66) ^ S64(1,  93);                                        \
    T(2) = S64(2,  69) ^ S64(2,  84);                                        \
    T(3) = S64(3,  66) ^ S96(3, 111);                                        \
                                                                             \
    Z(T(1) ^ T(2) ^ T(3));                                                   \
                                                                             \
    T(1) ^= (S64(1,  91) & S64(1,  92)) ^ S64(2,  78);                       \
    T(2) ^= (S64(2,  82) & S64(2,  83)) ^ S64(3,  87);                       \
    T(3) ^= (S96(3, 109) & S96(3, 110)) ^ S64(1,  69);                       \
  } while (0)

#define ROTATE()                                                             \
  do {                                                                       \
    S(1, 3) = S(1, 2); S(1, 2) = S(1, 1); S(1, 1) = T(3);                    \
    S(2, 3) = S(2, 2); S(2, 2) = S(2, 1); S(2, 1) = T(1);                    \
    S(3, 4) = S(3, 3); S(3, 3) = S(3, 2); S(3, 2) = S(3, 1); S(3, 1) = T(2); \
  } while (0)

#define LOAD(s)\
do {\
   S(1,1) = __mem((s),tID,0,nr_streams); S(2,1) = __mem((s),tID,3,nr_streams);\
   S(1,2) = __mem((s),tID,1,nr_streams); S(2,2) = __mem((s),tID,4,nr_streams);\
   S(1,3) = __mem((s),tID,2,nr_streams); S(2,3) = __mem((s),tID,5,nr_streams);\
\
   S(3,1) = __mem((s),tID,6,nr_streams); S(3,3) = __mem((s),tID,8,nr_streams);\
   S(3,2) = __mem((s),tID,7,nr_streams); S(3,4) = __mem((s),tID,9,nr_streams);\
} while(0)

#define STORE(s)\
do {\
   __mem((s),tID,0,nr_streams) = S(1,1); __mem((s),tID,3,nr_streams) = S(2,1);\
   __mem((s),tID,1,nr_streams) = S(1,2); __mem((s),tID,4,nr_streams) = S(2,2);\
   __mem((s),tID,2,nr_streams) = S(1,3); __mem((s),tID,5,nr_streams) = S(2,3);\
\
   __mem((s),tID,6,nr_streams) = S(3,1); __mem((s),tID,8,nr_streams) = S(3,3);\
   __mem((s),tID,7,nr_streams) = S(3,2); __mem((s),tID,9,nr_streams) = S(3,4);\
} while(0)

__global__ void Trivium_keyivsetup(u32* g_s,
                                   u32 *keys, u32 key_size,
                                   u32 *ivs, u32 iv_size) {
   u32 tID=blockIdx.x*blockDim.x+threadIdx.x;
   u32 nr_streams=blockDim.x*gridDim.x;

   u32 s11, s12, s13; 
   u32 s21, s22, s23; 
   u32 s31, s32, s33, s34; 

   u32 key0,key1,key2;
   u32 iv0,iv1,iv2;

   /* read key and iv */
   /* assuming the 4-byte aligned key/iv is 0'ed out if not a multiple of 4-bytes */
   key0 = (key_size>0 )?__mem(keys,tID,0,nr_streams):0;
   key1 = (key_size>32)?__mem(keys,tID,1,nr_streams):0;
   key2 = (key_size>64)?__mem(keys,tID,2,nr_streams):0;

   iv0 = (iv_size>0 )?__mem(ivs,tID,0,nr_streams):0;
   iv1 = (iv_size>32)?__mem(ivs,tID,1,nr_streams):0;
   iv2 = (iv_size>64)?__mem(ivs,tID,2,nr_streams):0;


   /* load key and iv */
   S(1,1)=key0;
   S(1,2)=key1;
   S(1,3)=key2&0xffff;

   S(2,1)=iv0;
   S(2,2)=iv1;
   S(2,3)=iv2&0xffff;

   S(3,1)=0;
   S(3,2)=0;
   S(3,3)=0;
   S(3,4)=0x00007000;

#define Z(w)
   for(int i = 0; i < 4 * 9; ++i) {
      u32 t1, t2, t3;

      UPDATE();
      ROTATE();
   }

   STORE(g_s);

  
}

__global__ void Trivium_process_bytes(gSTREAM_action act, u32* g_s,
                                       u32 *buff, u32 nr_words) {
   u32 tID=blockIdx.x*blockDim.x+threadIdx.x;
   u32 nr_streams=blockDim.x*gridDim.x;

   u32 s11, s12, s13; 
   u32 s21, s22, s23; 
   u32 s31, s32, s33, s34; 

   LOAD(g_s);

#undef Z
#define Z(w) (output_word ^= (w))
   for(int w=0;w<nr_words;w++) {
      u32 t1, t2, t3;
      u32 output_word=0;

      if(act!=GEN_KEYSTREAM) {
         output_word=__mem(buff,tID,w,nr_streams);
      }

      UPDATE();
      ROTATE();

      __mem(buff,tID,w,nr_streams)=CH_ENDIANESS32(output_word);
   }

   STORE(g_s);

}
#endif
