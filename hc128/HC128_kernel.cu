#ifndef __HC128_KERNEL_CU__
#define __HC128_KERNEL_CU__

#define __mem(mm,i,j,N) ((mm)[(i)+(j)*(N)])
#define max(a,b) (((a)>(b))?(a):(b))
#define min(a,b) (((a)<(b))?(a):(b))
#define mod(a,b) (((a)>=(b))?((a)-(b)):((a)<0)?((a)+(b)):(a)) 

#define lprintf(...) ;


#define rotl32(v, n) \
     ((u32)((v) << (n)) | ((v) >> (32 - (n))))

#define rotr32(v,n) rotl32(v,32-(n))

#define DEFINE16(L) \
   u32 L##0, L##1, L##2, L##3, L##4, L##5, L##6, L##7, L##8, L##9, L##10, L##11, L##12, L##13, L##14, L##15;

#define T(L)   (__mem(g_T,tID,(L),nr_streams))
#define W(L)   __mem(s_W,threadIdx.x,mod((L),17),blockDim.x)

extern __shared__ __align__ (__alignof(void*)) u32 smem_cache[];

#define P(L) (__mem(g_P,tID,(L),nr_streams))
#define Q(L) (__mem(g_Q,tID,(L),nr_streams))

#define f1(x) ((rotr32((x), 7)) ^ (rotr32((x),18)) ^ ((x)>> 3))
#define f2(x) ((rotr32((x),17)) ^ (rotr32((x),19)) ^ ((x)>>10))
#define g1(x,y,z) ((rotr32((x),10)^rotr32((z),23))+rotr32((y), 8))
#define g2(x,y,z) ((rotl32((x),10)^rotl32((z),23))+rotl32((y), 8))
#define h1(x) (Q(((x)&0xff)) + Q((256+(((x)>>16)&0xff))))
#define h2(x) (P(((x)&0xff)) + P((256+(((x)>>16)&0xff))))

__global__ void HC128_keyivsetup(u32* g_P, u32* g_Q,
                                 u32 *keys, u32 key_size,
                                 u32 *ivs, u32 iv_size) {
   u32 tID=blockIdx.x*blockDim.x+threadIdx.x;
   u32 nr_streams=blockDim.x*gridDim.x;
   int i,j;
   u32* s_W=(u32*) smem_cache;

   u32 k0,k1,k2,k3;
   u32 i0,i1,i2,i3;

   k0 = __mem(keys,tID,0,nr_streams); i0 = __mem(ivs,tID,0,nr_streams);
   k1 = __mem(keys,tID,1,nr_streams); i1 = __mem(ivs,tID,1,nr_streams);
   k2 = __mem(keys,tID,2,nr_streams); i2 = __mem(ivs,tID,2,nr_streams);
   k3 = __mem(keys,tID,3,nr_streams); i3 = __mem(ivs,tID,3,nr_streams);

   W(0)=W(4)=k0;
   W(1)=W(5)=k1;
   W(2)=W(6)=k2;
   W(3)=W(7)=k3;

   W( 8)=W(12)=i0;
   W( 9)=W(13)=i1;
   W(10)=W(14)=i2;
   W(11)=W(15)=i3;

   for(j=16,i=16;i<256; j=mod((j+1),17), i++) {
      W(j) = f2(W(j-2)) + W(j-7) + f1(W(j-15)) + W(j-16) + i;
   }

   for(i=0;i<512; j=mod((j+1),17), i++) {
      P(i) = W(j) = f2(W(j-2)) + W(j-7) + f1(W(j-15)) + W(j-16) + i + 256;
   }

   for(i=0;i<512; j=mod((j+1),17), i++) {
      Q(i) = W(j) = f2(W(j-2)) + W(j-7) + f1(W(j-15)) + W(j-16) + i + 768;
   }

   for(i=0;i<512;i++) {
      int x = mod((i-  3),512),
          y = mod((i- 10),512),
          z = mod((i-511),512),
          w = mod((i- 12),512);
      P(i) = ( P(i) + g1(P(x),P(y),P(z)) ) ^ h1(P(w));
   }

   for(i=0;i<512;i++) {
      int x = mod((i-  3),512),
          y = mod((i- 10),512),
          z = mod((i-511),512),
          w = mod((i- 12),512);
      Q(i) = ( Q(i) + g2(Q(x),Q(y),Q(z)) ) ^ h2(Q(w));
   }

}

__global__ void HC128_process_bytes(gSTREAM_action act, u32* g_P, u32 *g_Q,
                                    u32 *buff, u32 nr_words_done, u32 nr_words) {
   u32 tID=blockIdx.x*blockDim.x+threadIdx.x;
   u32 nr_streams=blockDim.x*gridDim.x;

   for(int wordno=nr_words_done;wordno<nr_words;wordno++) {
      u32 output_word=0;

      if(act!=GEN_KEYSTREAM) {
         output_word=__mem(buff,tID,wordno,nr_streams);
      }

      int j = wordno & 0x1ff;
      int x = mod((j-  3),512),
          y = mod((j- 10),512),
          z = mod((j-511),512),
          w = mod((j- 12),512);
      if((wordno&0x3ff)<512) {
         P(j)+=g1(P(x),P(y),P(z));
         output_word=h1(P(w))^P(j);
      } else {
         Q(j)+=g2(Q(x),Q(y),Q(z));
         output_word=h2(Q(w))^Q(j);
      }

      __mem(buff,tID,wordno,nr_streams)=output_word;
   }

}
#endif
