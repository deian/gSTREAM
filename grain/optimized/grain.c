/*
 * OPTIMIZED IMPLEMENTATION OF STREAM CIPHER GRAIN VERSION 1
 *
 * Filename: grain.c
 *
 * Author:
 * Deian Stefan
 * email: stefan@cooper.edu
 *
 * Synopsis:
 *  This file contains functions that implement the
 *  stream cipher Grain. It also implements functions 
 *  specified by the ECRYPT API.
 *
 *  This implementaiton is based on Martin Hell's code.
 */

#include "grain.h"

#include <stdio.h>
void print_ctx(ECRYPT_ctx *ctx) {
   int i;
   for(i=0;i<3;i++) {
      printf("[0x%08x,0x%08x]\n",ctx->s[i],ctx->b[i]);
   }
}
static void print_state(ECRYPT_ctx* ctx) {
   int i,j;
   printf("\n");
   printf("s = [");
   for(i=0;i<3;i++) {
      for(j=0;j<((i!=2)?32:16);j++)  {
         printf("%d",(ctx->s[i]>>j)&1);
      }
   }
   printf("]\n");

   printf("b = [");
   for(i=0;i<3;i++) {
      for(j=0;j<((i!=2)?32:16);j++)  {
         printf("%d",(ctx->b[i]>>j)&1);
      }
   }
   printf("]\n");
   
}

void ECRYPT_init(void){}

#define S0 (ctx->s[0])
#define S1 (ctx->s[1])
#define S2 (ctx->s[2])

#define B0 (ctx->b[0])
#define B1 (ctx->b[1])
#define B2 (ctx->b[2])

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

#define h(x0,x1,x2,x3,x4)\
   ((x1)^(x4)^((x0)&(x3))^((x2)&(x3))^((x3)&(x4))^((x0)&(x1)&(x2))^((x0)&(x2)&(x3))^((x0)&(x2)&(x4))^((x1)&(x2)&(x4))^((x2)&(x3)&(x4)))

#define SHIFT_FSR(S)\
      do {\
         S##0=(S##0>>1)|(((S##1)&1)<<31);\
         S##1=(S##1>>1)|(((S##2)&1)<<31);\
         S##2=(S##2>>1);\
      } while(0)

#define COMBINE_TERMS

/*
 * Function: grain_keystream
 *
 * Synopsis
 *  Generates a new bit and updates the internal state of the cipher.
 */
u8 grain_keystream(ECRYPT_ctx* ctx) {
   u32 x0  = get3(S),
       x1  = get25(S),
       x2  = get46(S),
       x3  = get64(S),
       x4  = get63(B);

   u32 Z   = get1(B) ^ get2(B) ^ get4(B) ^ get10(B) ^ get31(B) ^ get43(B) ^ get56(B) ^ h(x0,x1,x2,x3,x4);
   u32 S80 = get62(S) ^ get51(S) ^ get38(S) ^ get23(S) ^ get13(S) ^ get0(S);

#if !defined(COMBINE_TERMS)

   u32 B80 =(get0(S)) ^ (get62(B)) ^ (get60(B)) ^ (get52(B)) ^ (get45(B)) ^ (get37(B)) ^ (get33(B)) ^ (get28(B)) ^ (get21(B))^
      (get14(B)) ^ (get9(B)) ^ (get0(B)) ^ (get63(B)&get60(B)) ^ (get37(B)&get33(B)) ^ (get15(B)&get9(B))^
      (get60(B)&get52(B)&get45(B)) ^ (get33(B)&get28(B)&get21(B)) ^ (get63(B)&get45(B)&get28(B)&get9(B))^
      (get60(B)&get52(B)&get37(B)&get33(B)) ^ (get63(B)&get60(B)&get21(B)&get15(B))^
      (get63(B)&get60(B)&get52(B)&get45(B)&get37(B)) ^ (get33(B)&get28(B)&get21(B)&get15(B)&get9(B))^
      (get52(B)&get45(B)&get37(B)&get33(B)&get28(B)&get21(B));

#else

   u32 B33_28_21 = (get33(B)&get28(B)&get21(B));   /* 3 */
   u32 B52_45_37 = (get52(B)&get45(B)&get37(B));   /* 2 */
   u32 B52_37_33 = (get52(B)&get37(B)&get33(B));   /* 2 */
   u32 B60_52_45 = (get60(B)&get52(B)&get45(B));   /* 2 */
   u32 B63_60 = (get63(B)&get60(B));	/* 3 */
   u32 B37_33 = (get37(B)&get33(B));	/* 3 */
   u32 B45_28 = (get45(B)&get28(B));	/* 2 */
   u32 B15_9  = (get15(B)&get9(B));	/* 2 */
   u32 B21_15 = (get21(B)&get15(B));	/* 2 */

   u32 B80 =(get0(S)) ^ (get62(B)) ^ (get60(B)) ^ (get52(B)) ^ (get45(B)) ^ (get37(B)) ^ (get33(B)) ^ (get28(B)) ^ (get21(B))^
      (get14(B)) ^ (get9(B)) ^ (get0(B)) ^ (B63_60) ^ (B37_33) ^ (B15_9)^
      (B60_52_45) ^ (B33_28_21) ^ (get63(B)&B45_28&get9(B))^
      (get60(B)&B52_37_33) ^ (B63_60&B21_15)^
      (B63_60&B52_45_37) ^ (B33_28_21&B15_9)^
      (B52_45_37&B33_28_21);
#endif



   SHIFT_FSR(S);
   SHIFT_FSR(B);
   set79(S,S80);
   set79(B,B80);

   return Z&1;
}


/* Functions for the ECRYPT API */

void ECRYPT_keysetup(
  ECRYPT_ctx* ctx, 
  const u8* key, 
  u32 keysize,                /* Key size in bits. */ 
  u32 ivsize)				  /* IV size in bits. */ 
{
	ctx->p_key=key;
	ctx->keysize=keysize;
	ctx->ivsize=ivsize;
}

/*
 * Function: ECRYPT_ivsetup
 *
 * Synopsis
 *  Load the key and perform initial clockings.
 *
 * Assumptions
 *  The key is 10 bytes and the IV is 8 bytes.
 *  
 */
void ECRYPT_ivsetup(
  ECRYPT_ctx* ctx, 
  const u8* iv)
{
   u32 outbit;
   int i;
   u8 *b=(u8*)ctx->b;
   u8 *s=(u8*)ctx->s;

   for(i=0;i<10;i++) 
      b[i]=ctx->p_key[i];

   for(i=0;i<ctx->ivsize/8;i++) 
      s[i]=iv[i];

   for(i=ctx->ivsize/8;i<10;i++)
      s[i]=0xff;


   /* do initial clockings */
   for (i=0;i<INITCLOCKS;++i) {
      outbit=grain_keystream(ctx);
      xor79(S,outbit);
      xor79(B,outbit);
   }
   //print_ctx(ctx);
}

/*
 * Function: ECRYPT_keystream_bytes
 *
 * Synopsis
 *  Generate keystream in bytes.
 *
 * Assumptions
 *  Bits are generated in order z0,z1,z2,...
 *  The bits are stored in a byte in order:
 *  
 *  lsb of keystream[0] = z0
 *  ...
 *  msb of keystream[0] = z7
 *  ...
 *  lsb of keystream[1] = z8
 *  ...
 *  msb of keystream[1] = z15
 *  ...
 *  ...
 *  ...
 *  Example: The bit keystream: 10011100 10110011 ..
 *  corresponds to the byte keystream: 39 cd ..
 */
void ECRYPT_keystream_bytes(
  ECRYPT_ctx* ctx, 
  u8* keystream, 
  u32 msglen)
{
	u32 i,j;
	for (i = 0; i < msglen; ++i) {
		keystream[i]=0;
		for (j = 0; j < 8; ++j) {
			keystream[i]|=(grain_keystream(ctx)<<j);
		}
	}
}
void ECRYPT_encrypt_bytes(
  ECRYPT_ctx* ctx, 
  const u8* plaintext, 
  u8* ciphertext, 
  u32 msglen)
{
	u32 i,j;
	u8 k;
	for (i = 0; i < msglen; ++i) {
		k=0;
		for (j = 0; j < 8; ++j) {	
			k|=(grain_keystream(ctx)<<j);
		}
		ciphertext[i]=plaintext[i]^k;
	}
}

void ECRYPT_decrypt_bytes(
  ECRYPT_ctx* ctx, 
  const u8* ciphertext, 
  u8* plaintext, 
  u32 msglen)
{
	u32 i,j;
	u8 k=0;
	for (i = 0; i < msglen; ++i) {
		k=0;
		for (j = 0; j < 8; ++j) {
			k|=(grain_keystream(ctx)<<j);
		}
		plaintext[i]=ciphertext[i]^k;
	}
}
