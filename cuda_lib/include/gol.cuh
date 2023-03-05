#ifndef _GOL_CUH_
#define _GOL_CUH_

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

constexpr int BLOCK_X = 32;  // the size of the x dimension of a block
constexpr int BLOCK_Y = 16; // the size of the y dimension of a block

__global__ void
nextGen(char* d_life, char* d_life_copy, const int rows, const int columns);

__global__ void
nextGenInner(char* d_life, char* d_life_copy, const int rows, const int columns);

__global__ void
nextGenOuter(char* d_life, char* d_life_copy, const int rows, const int columns); 

__global__ void
packCols(char* d_life, char* leftHaloColumn, char* rightHaloColumn, int rows, int columns); 

__global__ void
unpackCols(char* d_life, char* leftHaloColumn, char* rightHaloColumn, int rows, int columns); 

extern "C" void
nextGeneration(char* d_life, char* d_life_copy, int rows, int columns);

extern "C" void
nextGenerationInner(char* d_life, char* d_life_copy, int rows, int columns);

extern "C" void
nextGenerationOuter(char* d_life, char* d_life_copy, int rows, int columns);

extern "C" cudaError_t
initializeOnDevice(char** d_life, char** d_life_copy, char** leftHaloColumnBufferSend, char** leftHaloColumnBufferRecv, char** rightHaloColumnBufferSend, char** rightHaloColumnBufferRecv, char** h_life, int rows, int columns);

extern "C" void
getGPUCount(int* numOfGPUs);

extern "C" void
packHaloColumns(char* d_life, char* leftHaloColumn, char* rightHaloColumn, int rows, int columns); 

extern "C" void
unpackHaloColumns(char* d_life, char* leftHaloColumn, char* rightHaloColumn, int rows, int columns); 
#endif
