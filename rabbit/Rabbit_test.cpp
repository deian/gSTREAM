#include "gSTREAM.h"
#include "gSTREAM_test.h"

int main(void) {
   do_test(0
          ,2,128,680
          ,GEN_KEYSTREAM,16,8
          ,1024);
   return 0;
}
