#ifndef __MICKEY_KERNEL_CU__
#define __MICKEY_KERNEL_CU__

#define __mem(mm,i,j,N) ((mm)[(i)+(j)*(N)])
#define max(a,b) (((a)>(b))?(a):(b))
#define min(a,b) (((a)<(b))?(a):(b))

__device__ void CLOCK_R(u32 *oR0, u32 *oR1, u32 *oR2, u32 *oR3, int input_bit, int control_bit) {
    u32 R0=*oR0, R1=*oR1, R2=*oR2, R3=*oR3;
    int Feedback_bit;
    int Carry0, Carry1, Carry2;
    Feedback_bit = ((R3 >> 3) & 1) ^ input_bit;
    Carry0 = (R0 >> 31) & 1;
    Carry1 = (R1 >> 31) & 1;
    Carry2 = (R2 >> 31) & 1;

    if (control_bit) {
        R0 ^= (R0 << 1);
        R1 ^= (R1 << 1) ^ Carry0;
        R2 ^= (R2 << 1) ^ Carry1;
        R3 ^= (R3 << 1) ^ Carry2;
    } else {
        R0 = (R0 << 1);
        R1 = (R1 << 1) ^ Carry0;
        R2 = (R2 << 1) ^ Carry1;
        R3 = (R3 << 1) ^ Carry2;
    }

    if (Feedback_bit) {
        R0 ^= 0x1279327b;
        R1 ^= 0xb5546660;
        R2 ^= 0xdf87818f;
        R3 ^= 0x00000003;
    }
    *oR0=R0;
    *oR1=R1;
    *oR2=R2;
    *oR3=R3;
}

__device__ void CLOCK_S(u32 *oS0, u32 *oS1, u32 *oS2, u32 *oS3, int input_bit, int control_bit) {
    u32 S0=*oS0, S1=*oS1, S2=*oS2, S3=*oS3;
    int Feedback_bit;
    int Carry0, Carry1, Carry2;


    Feedback_bit = ((S3 >> 3) & 1) ^ input_bit;
    Carry0 = (S0 >> 31) & 1;
    Carry1 = (S1 >> 31) & 1;
    Carry2 = (S2 >> 31) & 1;

    S0 = (S0 << 1) ^ ((S0 ^ 0x6aa97a30) & ((S0 >> 1) ^ (S1 << 31) ^ 0xdd629e9a) & 0xfffffffe);
    S1 = (S1 << 1) ^ ((S1 ^ 0x7942a809) & ((S1 >> 1) ^ (S2 << 31) ^ 0xe3a21d63)) ^ Carry0;
    S2 = (S2 << 1) ^ ((S2 ^ 0x057ebfea) & ((S2 >> 1) ^ (S3 << 31) ^ 0x91c23dd7)) ^ Carry1;
    S3 = (S3 << 1) ^ ((S3 ^ 0x00000006) & ((S3 >> 1) ^ 0x00000001) & 0x7) ^ Carry2;

    if (Feedback_bit) {
        if (control_bit) {
            S0 ^= 0x4c8cb877;
            S1 ^= 0x4911b063;
            S2 ^= 0x40fbc52b;
            S3 ^= 0x00000008;
        } else {
            S0 ^= 0x9ffa7faf;
            S1 ^= 0xaf4a9381;
            S2 ^= 0x9cec5802;
            S3 ^= 0x00000001;
        }
    }
    *oS0=S0;
    *oS1=S1;
    *oS2=S2;
    *oS3=S3;
}

#define CLOCK_KG(Keystream_bit,R0,R1,R2,R3,S0,S1,S2,S3,mixing,input_bit)                                      \
do {                                                                                            \
    int control_bit_r;                                                                          \
    int control_bit_s;                                                                          \
                                                                                                \
    (Keystream_bit) = ((R0) ^ (S0)) & 1;                                                        \
    control_bit_r = (((S1) >> 2) ^ ((R2) >> 3)) & 1;                                            \
    control_bit_s = (((R1) >> 1) ^ ((S2) >> 3)) & 1;                                            \
                                                                                                \
    if((mixing)) {                                                                              \
        CLOCK_R(&(R0), &(R1), &(R2), &(R3), (((S1) >> 18) & 1) ^ (input_bit), control_bit_r);   \
    } else {                                                                                    \
        CLOCK_R(&(R0), &(R1), &(R2), &(R3), (input_bit), control_bit_r);                        \
    }                                                                                           \
                                                                                                \
    CLOCK_S(&(S0), &(S1), &(S2), &(S3), (input_bit), control_bit_s);                            \
                                                                                                \
} while(0)

__global__ void MICKEY_keyivsetup(u32* g_r, u32* g_s,
                                  u32 *keys, u32 key_size,
                                  u32 *ivs, u32 iv_size) {
   u32 tID=blockIdx.x*blockDim.x+threadIdx.x;
   u32 nr_streams=blockDim.x*gridDim.x;

   u32 r0,r1,r2,r3,
       s0,s1,s2,s3;
   int Keystream_bit;

   u32 sub_keyiv;
   int ivkey_bit;
   int ivkey_no;
   int i;

   r0=0; s0=0;
   r1=0; s1=0;
   r2=0; s2=0;
   r3=0; s3=0;

   ivkey_no=0;
   while(iv_size>0) {
      sub_keyiv = __mem(ivs,tID,ivkey_no++,nr_streams);
      for(i=0;i<min(iv_size,32);i++) {
         ivkey_bit=(sub_keyiv&0x80000000)?1:0;
         CLOCK_KG(Keystream_bit,r0,r1,r2,r3,s0,s1,s2,s3,1,ivkey_bit);
         sub_keyiv<<=1;
      }
      iv_size=max((int)(iv_size-32),0);
   }

   ivkey_no=0;
   while(key_size>0) {
      sub_keyiv = __mem(keys,tID,ivkey_no++,nr_streams);
      for(i=0;i<min(key_size,32);i++) {
         ivkey_bit=(sub_keyiv&0x80000000)?1:0;
         CLOCK_KG(Keystream_bit,r0,r1,r2,r3,s0,s1,s2,s3,1,ivkey_bit);
         sub_keyiv<<=1;
      }
      key_size=max((int)(key_size-32),0);
   }

   for(i=0;i<100;i++) {
         CLOCK_KG(Keystream_bit,r0,r1,r2,r3,s0,s1,s2,s3,1,0);
   }
   __mem(g_r,tID,0,nr_streams)=r0; __mem(g_s,tID,0,nr_streams)=s0;
   __mem(g_r,tID,1,nr_streams)=r1; __mem(g_s,tID,1,nr_streams)=s1;
   __mem(g_r,tID,2,nr_streams)=r2; __mem(g_s,tID,2,nr_streams)=s2;
   __mem(g_r,tID,3,nr_streams)=r3; __mem(g_s,tID,3,nr_streams)=s3;

}

__global__ void MICKEY_process_bytes(gSTREAM_action act,u32* g_r, u32* g_s,
                                       u32 *buff, u32 nr_words) {
   u32 tID=blockIdx.x*blockDim.x+threadIdx.x;
   u32 nr_streams=blockDim.x*gridDim.x;

   u32 r0,r1,r2,r3,
       s0,s1,s2,s3;

   int Keystream_bit;

   r0=__mem(g_r,tID,0,nr_streams); s0=__mem(g_s,tID,0,nr_streams);
   r1=__mem(g_r,tID,1,nr_streams); s1=__mem(g_s,tID,1,nr_streams);
   r2=__mem(g_r,tID,2,nr_streams); s2=__mem(g_s,tID,2,nr_streams);
   r3=__mem(g_r,tID,3,nr_streams); s3=__mem(g_s,tID,3,nr_streams);

   for(int w=0;w<nr_words;w++) {
      u32 output_word=0;

      if(act!=GEN_KEYSTREAM) {
         output_word=__mem(buff,tID,w,nr_streams);
      }

#pragma unroll 32
      for(int i=0;i<32;i++) {
         CLOCK_KG(Keystream_bit,r0,r1,r2,r3,s0,s1,s2,s3,0,0);
         output_word ^= Keystream_bit << (31-i);
      }

      __mem(buff,tID,w,nr_streams)=CH_ENDIANESS32(output_word);
   }
   __mem(g_r,tID,0,nr_streams)=r0; __mem(g_s,tID,0,nr_streams)=s0;
   __mem(g_r,tID,1,nr_streams)=r1; __mem(g_s,tID,1,nr_streams)=s1;
   __mem(g_r,tID,2,nr_streams)=r2; __mem(g_s,tID,2,nr_streams)=s2;
   __mem(g_r,tID,3,nr_streams)=r3; __mem(g_s,tID,3,nr_streams)=s3;

}
#endif
