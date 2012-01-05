#include "gSTREAM.h"
#include "gSTREAM_test.h"

int main(void) {
   do_test(0
          ,2,128,680
          ,GEN_KEYSTREAM,10,8
          ,2048);
   return 0;
}
