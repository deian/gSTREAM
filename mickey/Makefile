################################################################################

TARGET=MICKEY
# Add source files here
EXECUTABLE	:= $(TARGET)
# CUDA source files (compiled with cudacc)
CUFILES		:= gSTREAM.cu
# CUDA dependency files
CU_DEPS		:= gSTREAM.h 

# C/C++ source files (compiled with gcc / c++)
CCFILES		:= gSTREAM_test.cpp  MICKEY_test.cpp

keep            =1
verbose         =1
NVCCFLAGS := -Xptxas -v
#--maxrregcount 16



################################################################################
# Rules and targets

#include ../../common/common.mk
include /home/d/NVIDIA_GPU_Computing_SDK/C/common/common.mk
profile:
	zsh prof.sh
	./release/aes_c &> output
	tail log.log
	tail output
