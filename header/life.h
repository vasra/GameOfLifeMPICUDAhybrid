#pragma once
#include "device_launch_parameters.h"

void Initial_state(int size, char* first_generation, char* first_generation_copy);
void Print_grid(int size, char* life);
extern "C" float GameOfLife(const int size, char* life, char* life_copy, int dimGr, dim3 dimBl, int generations);
__global__ void nextGen(char* d_life, char* d_life_copy, const int size, int nblocks, dim3 dimBl);