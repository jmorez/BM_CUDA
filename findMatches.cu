/*  ###########################DESCRIPTION###############################
 *  Written by XXXXXXX (University of XXXXXX) as a part of a bachelor's 
 *  thesis that uses a blockmatching algorithm  to gather a statistical 
 *  population for denoising single pixels in an image.
 */

/*  This function contains the findMatches CUDA kernel that will find 
 *  matches for every pixel in an image of size M by N (rows x columns).
 *  The algorithm is based on the following paper:
 *  http://www.mia.uni-saarland.de/Publications/zimmer-lnla08.pdf
 */



 /* ###########################STYLE NOTES###############################
  * Device variables will have the prefix d_, no prefix implies a host 
  * variable.
  *
  * Throughout this code, I will insert footnotes inside comments of the 
  * format (#) which -unsurprisingly- can be found at the bottom. This is 
  * first of all to keep the code compact, but also to allow both the 
  * reader as myself to understand this code, i.e. "why 
  * use datatype X", "why do loop Y like this", "why is thisindex Z minus 
  * one", ...  
 */

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "mxGPUImage.h"

#include <math.h>


/*  Kernel code, using floats because doubles can drastically hurt perfor-
 *  mance.
 */
void __global__ findMatches(const mxGPUArray* d_img, const int* imgSize, const int blocksize){
    //Array coordinates of the reference block. 
    const int i = blockDim.x*blockIdx.x+threadIdx.x;
    const int j = blockDim.y*blockIdx.y+threadIdx.y;
    if (i < imgSize[0] && j < imgSize[1]){
        //Fetch the reference block
       
        
        //CompareBlocks (ref, [search_window])
        
        
    }
}

//I'll need a way to fetch square regions of an image easily
void __device__ getRegionAroundPixel(   const mxGPUArray* d_img,
                                        const mxGPUArray* d_result,
                                        const int radius,
                                        const int i, 
                                        const int j){
    /*
    const int min_i=i-radius;
    const int max_i=i+radius;
    const int min_j=j-radius;
    const int max_j=j+radius;
    */
      
}

/* Call in matlab like this:
[plhs[0],plhs[1],plhs[...],plhs[nrhs-1]]=filename(prhs[0],prhs[1],prhs[...],prhs[nrhs-1])
*/
void mexFunction(   int nlhs, mxArray *plhs[],
                    int nrhs, mxArray const *prhs[]){
    /* prhs argument explanation:
     *plhs[0]: mxArray that contains the image. (1)
     */

    //Variable declarations
    const mxGPUArray* A;
    dim3 blocksPerGrid;

    dim3 threadsPerBlock;

    //Input verification
    
    
    //Create the image array on the GPU. Edit: fuck this datatype, CUDA arrays are
    //more useful
    A=mxGPUCreateFromMxArray(prhs[0]);
    const mwSize* img_size =  mxGPUGetDimensions(A);
    const size_t M=img_size[0];
    const size_t N=img_size[2];

    
    //Initialize MathWorks GPU API. 
    mxInitGPU();
     
    //Kernel parameters
	/*Figure out grid layout. (1)
	*/
	cudaDeviceProp device;
	cudaGetDeviceProperties(&device,0);
	const int MaxThreadsPerBlock=device.maxThreadsPerBlock;

	threadsPerBlock.x=(size_t)sqrt((double) MaxThreadsPerBlock);
	threadsPerBlock.y=(size_t)sqrt((double) MaxThreadsPerBlock);
   
    blocksPerGrid.x=(size_t)(M-1)/threadsPerBlock.x+1;
    blocksPerGrid.y=(size_t)(N-1)/threadsPerBlock.y+1;
    blocksPerGrid.z=1;
    
    //findMatches<<<blocksPerGrid,threadsPerBlock>>>(A,M,N);
    
    mxGPUImage test;
    
    mxGPUDestroyGPUArray(A);
}


    


/*####################### FOOTNOTES ##################################
 * (1)  We'll use a 2D grid where each thread corresponds with
 *      one pixel. We'll go for 1024 threads per block, which for a 2.1 CC 
 *      device gives us 67% occupancy.
 */
