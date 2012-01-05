/*
 * REFERENCE IMPLEMENTATION OF STREAM CIPHER GRAIN VERSION 1
 *
 * Filename: testvectors.c
 *
 * Author:
 * Martin Hell
 * Dept. of Information Technology
 * P.O. Box 118
 * SE-221 00 Lund, Sweden,
 * email: martin@it.lth.se
 *
 * Synopsis:
 *    Generates testvectors from the reference implementation of Grain Version 1.
 *
 */


#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include "grain.h"

void printData(u8 *key, u8 *IV, u8 *ks) {
	u32 i;
	printf("\n\nkey:        ");
	for (i=0;i<10;++i) printf("%02x",(int)key[i]);
	printf("\nIV :        ");
	for (i=0;i<8;++i) printf("%02x",(int)IV[i]);
	printf("\nkeystream:  ");
	for (i=0;i<KS_SIZE;++i) printf("%02x",(int)ks[i]);
}

void testvectors() {
	
   int i;
   ECRYPT_ctx ctx;
   u8 key[10] = {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
      IV[8] = {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
      *ks;

   if(!(ks=malloc(KS_SIZE*sizeof(u8)))) { 
      fprintf(stderr, "Allocating ks failed: %s\n",strerror(errno));
      exit(-1);
   }

   for(i=0;i<10;i++) key[i]=rand();
   for(i=0;i<8;i++) IV[i]=rand();

   ECRYPT_keysetup(&ctx,key,80,64);
   ECRYPT_ivsetup(&ctx,IV);
   ECRYPT_keystream_bytes(&ctx,ks,KS_SIZE);

#ifdef VERBOSE
   printData(key,IV,ks);
   printf("\n");
#endif

}

int main(int argc, char **argv) {	
	testvectors();
	return 0;
}


