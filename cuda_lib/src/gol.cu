#include <gol.cuh>
#include <algorithm>

extern "C" void
getGPUCount(int* numOfGPUs) {
   cudaGetDeviceCount(numOfGPUs);
}

__global__ void
nextGen(char* d_life, char* d_life_copy, const int rows, const int columns) {
   // Shared memory grid
   __shared__ char sgrid[BLOCK_Y][BLOCK_X];

   int X = blockIdx.x * (blockDim.x - 2) + threadIdx.x;
   int Y = blockIdx.y * (blockDim.y - 2) + threadIdx.y;

   // The global ID of the thread in the grid
   int threadIdGlobal = columns * Y + X;

   // The local coordinates of the thread in the block
   int x = threadIdx.x;
   int y = threadIdx.y;

   int neighbours = 0;

   if (X <= columns + 1 && Y <= rows + 1) {
      sgrid[y][x] = d_life[threadIdGlobal];
   }

   __syncthreads();

   // If the thread does not correspond to a halo element inside the block, then calculate its neighbours
   if (X > 0 && X < columns - 1 && Y > 0 && Y < rows - 1 &&
       x > 0 && x < blockDim.x - 1 && y > 0 && y < blockDim.y - 1) {
      
      neighbours = sgrid[y - 1][x - 1] + sgrid[y - 1][x]  + sgrid[y - 1][x + 1] +
                   sgrid[y][x - 1]     + /* you are here */ sgrid[y][x + 1]     +
                   sgrid[y + 1][x - 1] + sgrid[y + 1][x]  + sgrid[y + 1][x + 1];
      
      if ((2 == neighbours && 1 == sgrid[y][x]) || (3 == neighbours)) {
         sgrid[y][x] = 1;
      } else {
         sgrid[y][x] = 0;
      }
      d_life_copy[threadIdGlobal] = sgrid[y][x];
   }
}

extern "C" void
nextGeneration(char* d_life, char* d_life_copy, int rows, int columns) {
   // The layout of the threads in the block
   dim3 threadsInBlock{ BLOCK_X, BLOCK_Y, 1 };

   // Check is necessary to avoid division by zero
   // The layout of the blocks in the grid. We subtract 2 from each
   // coordinate, to compensate for the halo rows and columns of each block
   unsigned int gridX = static_cast<int>(ceil(columns / static_cast<float>(BLOCK_X - 2)));
   unsigned int gridY = static_cast<int>(ceil(rows / static_cast<float>(BLOCK_Y - 2)));
   dim3 gridDims{ gridX, gridY, 1 };
   nextGen <<<gridDims, threadsInBlock>>> (d_life, d_life_copy, rows, columns);
}

__global__ void
nextGenInner(char* d_life, char* d_life_copy, const int rows, const int columns) {
   // Shared memory grid
   __shared__ char sgrid[BLOCK_Y][BLOCK_X];

   int X = blockIdx.x * (blockDim.x - 2) + threadIdx.x;
   int Y = blockIdx.y * (blockDim.y - 2) + threadIdx.y;

   // The global ID of the thread in the grid
   int threadIdGlobal = columns * Y + X;

   // The local coordinates of the thread in the block
   int x = threadIdx.x;
   int y = threadIdx.y;

   int neighbours = 0;

   // Since we are calculating only the inner elements, we need threads
   // with ID greater than the halos and their immediate
   // neighbouring rows/columns
   if (X > 1 && X < columns - 2 && Y > 1 && Y < rows - 2) {
      sgrid[y][x] = d_life[threadIdGlobal];
   }

   __syncthreads();

   // If the thread does not correspond to a halo element inside the block, then calculate its neighbours
   if (X > 1 && X < columns - 2 && Y > 1 && Y < rows - 2 &&
       x > 0 && x < blockDim.x - 1 && y > 0 && y < blockDim.y - 1) {
      
      neighbours = sgrid[y - 1][x - 1] + sgrid[y - 1][x]  + sgrid[y - 1][x + 1] +
                   sgrid[y][x - 1]     + /* you are here */ sgrid[y][x + 1]     +
                   sgrid[y + 1][x - 1] + sgrid[y + 1][x]  + sgrid[y + 1][x + 1];
      
      if ((2 == neighbours && 1 == sgrid[y][x]) || (3 == neighbours)) {
         sgrid[y][x] = 1;
      } else {
         sgrid[y][x] = 0;
      }
      d_life_copy[threadIdGlobal] = sgrid[y][x];
   }
}

__global__ void
nextGenOuter(char* d_life, char* d_life_copy, const int rows, const int columns) {
   // Shared memory grid
   __shared__ char sgrid[BLOCK_Y][BLOCK_X];

   int X = blockIdx.x * (blockDim.x - 2) + threadIdx.x;
   int Y = blockIdx.y * (blockDim.y - 2) + threadIdx.y;

   // The global ID of the thread in the grid
   int threadIdGlobal = columns * Y + X;

   // The local coordinates of the thread in the block
   int x = threadIdx.x;
   int y = threadIdx.y;

   int neighbours = 0;

   if (X <= columns + 1 && Y <= rows + 1) {
      sgrid[y][x] = d_life[threadIdGlobal];
   }

   __syncthreads();

   // Since we are calculating only the outer elements, we need threads
   // with ID that corresponds to the first/last real rows, as well
   // threads with ID that corresponds to the first/last real columns
   bool belongsToFirstLastRow    = (X >= 1 && X <= columns - 2) && (Y == 1 || Y == rows - 2);
   bool belongsToFirstLastColumn = (X == 1 || X == columns - 2) && (Y >= 1 && Y <= rows - 2); 

   if ((belongsToFirstLastRow || belongsToFirstLastColumn) &&
       x > 0 && x < blockDim.x - 1 && y > 0 && y < blockDim.y - 1) {
      neighbours = sgrid[y - 1][x - 1] + sgrid[y - 1][x]  + sgrid[y - 1][x + 1] +
                   sgrid[y][x - 1]     + /* you are here */ sgrid[y][x + 1]     +
                   sgrid[y + 1][x - 1] + sgrid[y + 1][x]  + sgrid[y + 1][x + 1];
      if ((2 == neighbours && 1 == sgrid[y][x]) || (3 == neighbours)) {
         sgrid[y][x] = 1;
      } else {
         sgrid[y][x] = 0;
      }
      d_life_copy[threadIdGlobal] = sgrid[y][x];
   }
}

__global__ void
packCols(char* d_life, char* leftHaloColumn, char* rightHaloColumn, int rows, int columns) {
   int X = blockIdx.x * blockDim.x + threadIdx.x;
   int Y = blockIdx.y * blockDim.y + threadIdx.y;

   bool isOnLeftRealmostColumn = ((X == 1) && (Y > 0) && (Y < (rows - 1)));
   bool isOnRightRealColumn = ((X == (columns - 2)) && (Y > 0) && (Y < (rows - 1)));

   if (isOnLeftRealmostColumn) {  
      *(leftHaloColumn + Y - 1) = *(d_life + (Y * columns) + 1);
   } else if (isOnRightRealColumn) {
      *(rightHaloColumn + Y - 1) = *(d_life + (Y * columns) + columns - 2);
   }
}


__global__ void
unpackCols(char* d_life, char* leftHaloColumn, char* rightHaloColumn, int rows, int columns) {
   int X = blockIdx.x * blockDim.x + threadIdx.x;
   int Y = blockIdx.y * blockDim.y + threadIdx.y;

   bool isOnLeftRealmostColumn = ((X == 1) && (Y > 0) && (Y < (rows - 1)));
   bool isOnRightRealColumn = ((X == (columns - 2)) && (Y > 0) && (Y < (rows - 1)));

   if (isOnLeftRealmostColumn) {  
       *(d_life + (Y * columns)) = *(leftHaloColumn + Y - 1);
   } else if (isOnRightRealColumn) {
       *(d_life + (Y * columns) + columns - 1) = *(rightHaloColumn + Y - 1);
   }
}

extern "C" void
nextGenerationInner(char* d_life, char* d_life_copy, int rows, int columns) {
   // The layout of the threads in the block
   dim3 threadsInBlock{ BLOCK_X, BLOCK_Y, 1 };

   // Check is necessary to avoid division by zero
   // The layout of the blocks in the grid. We subtract 2 from each
   // coordinate, to compensate for the halo rows and columns of each block

   unsigned int gridX = static_cast<int>(ceil((columns - 2) / static_cast<float>(BLOCK_X - 2)));
   unsigned int gridY = static_cast<int>(ceil((rows - 2) / static_cast<float>(BLOCK_Y - 2)));
   dim3 gridDims{ gridX, gridY, 1 };
   nextGenInner <<<gridDims, threadsInBlock>>> (d_life, d_life_copy, rows, columns);
}

extern "C" void
nextGenerationOuter(char* d_life, char* d_life_copy, int rows, int columns) {
   // The layout of the threads in the block
   dim3 threadsInBlock{ BLOCK_X, BLOCK_Y, 1 };

   // Check is necessary to avoid division by zero
   // The layout of the blocks in the grid. We subtract 2 from each
   // coordinate, to compensate for the halo rows and columns of each block
   unsigned int gridX = static_cast<int>(ceil((columns - 2) / static_cast<float>(BLOCK_X - 2)));
   unsigned int gridY = static_cast<int>(ceil((rows - 2) / static_cast<float>(BLOCK_Y - 2)));
   dim3 gridDims{ gridX, gridY, 1 };
   nextGenOuter <<<gridDims, threadsInBlock>>> (d_life, d_life_copy, rows, columns);
}

extern "C" cudaError_t
initializeOnDevice(char** d_life, char** d_life_copy, char** leftHaloColumnBufferSend, char** leftHaloColumnBufferRecv, char** rightHaloColumnBufferSend, char** rightHaloColumnBufferRecv, char** h_life, int rows, int columns) {
   cudaError_t err = cudaSuccess;

   err = cudaMalloc((void**)&(*d_life), rows * columns * sizeof(char));
   if (cudaSuccess != err) {
      fprintf(stderr, "Could not allocate CUDA memory, with error code %d\n", err);
      return err;
   }

   err = cudaMemcpy(*d_life, *h_life, rows * columns * sizeof(char), cudaMemcpyHostToDevice);
   if (cudaSuccess != err) {
      fprintf(stderr, "Could not copy to GPU memory, with error code %d\n", err);
      return err;
   }

   err = cudaMalloc((void**)&(*d_life_copy), rows * columns * sizeof(char));
   if (cudaSuccess != err) {
      fprintf(stderr, "Could not allocate CUDA memory, with error code %d\n", err);
      return err;
   }

   err = cudaMemcpy(*d_life_copy, *h_life, rows * columns * sizeof(char), cudaMemcpyHostToDevice);
   if (cudaSuccess != err) {
      fprintf(stderr, "Could not copy to GPU memory, with error code %d\n", err);
      return err;
   }

   err = cudaMalloc((void**)&(*leftHaloColumnBufferSend), (rows  - 2) * sizeof(char));
   if (cudaSuccess != err) {
      fprintf(stderr, "Could not allocate CUDA memory, with error code %d\n", err);
      return err;
   }

   err = cudaMalloc((void**)&(*rightHaloColumnBufferSend), (rows - 2) * sizeof(char));
   if (cudaSuccess != err) {
      fprintf(stderr, "Could not allocate CUDA memory, with error code %d\n", err);
      return err;
   }

   err = cudaMalloc((void**)&(*leftHaloColumnBufferRecv), (rows  - 2) * sizeof(char));
   if (cudaSuccess != err) {
      fprintf(stderr, "Could not allocate CUDA memory, with error code %d\n", err);
      return err;
   }

   err = cudaMalloc((void**)&(*rightHaloColumnBufferRecv), (rows - 2) * sizeof(char));
   if (cudaSuccess != err) {
      fprintf(stderr, "Could not allocate CUDA memory, with error code %d\n", err);
      return err;
   }

   return err;
} 

extern "C" void
packHaloColumns(char* d_life, char* leftHaloColumn, char* rightHaloColumn, int rows, int columns) {
   unsigned int gridX = static_cast<int>(ceil(columns / static_cast<float>(BLOCK_X)));
   unsigned int gridY = static_cast<int>(ceil(rows / static_cast<float>(BLOCK_Y)));

   dim3 blockDims{ BLOCK_X, BLOCK_Y, 1 };
   dim3 gridDims{ gridX, gridY, 1 };

   packCols<<<gridDims, blockDims>>>(d_life, leftHaloColumn, rightHaloColumn, rows, columns);
}


extern "C" void
unpackHaloColumns(char* d_life, char* leftHaloColumn, char* rightHaloColumn, int rows, int columns) {
   unsigned int gridX = static_cast<int>(ceil(columns  / static_cast<float>(BLOCK_X - 2)));
   unsigned int gridY = static_cast<int>(ceil(rows / static_cast<float>(BLOCK_Y - 2)));

   dim3 blockDims{ BLOCK_X, BLOCK_Y, 1 };
   dim3 gridDims{ gridX, gridY, 1 };

   unpackCols<<<gridDims, blockDims>>>(d_life, leftHaloColumn, rightHaloColumn, rows, columns);
}
