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
#include <math.h>

/*  Kernel code, using floats because doubles can drastically hurt perfor-
 *  mance.
 */
void __global__ findMatches(const float* d_img, const int M, const int N){
    //Array coordinates of the reference block. 
    const int i = blockDim.x*blockIdx.x+threadIdx.x;
    const int j = blockDim.y*blockIdx.y+threadIdx.y;
    if (i < M && j < N){
        // Do stuff...
    }
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
    
    //Create the image array on the GPU
    A=mxGPUCreateFromMxArray(prhs[0]);
    const mwSize* img_size =  mxGPUGetDimensions(A);
    const int M=img_size[0];
    const int N=img_size[1];
    
    mexPrintf("\n %i %i \n",M,N);
    
    //Initialize MathWorks GPU API. 
    mxInitGPU();
     
    //Kernel parameters
	/*Figure out grid layout. (1)
	*/
	cudaDeviceProp device;
	cudaGetDeviceProperties(&device,0);
	const int MaxThreadsPerBlock=device.maxThreadsPerBlock;
	dim3 BlockDim;
	BlockDim.x=sqrt((double) MaxThreadsPerBlock);
	BlockDim.y=sqrt((double) MaxThreadsPerBlock);
    
    
    
}


    


/*####################### FOOTNOTES ##################################
 * (1)  We'll use a 2D grid where each thread corresponds with
 *      one pixel. We'll go for 1024 threads per block, which for a 2.1 CC 
 *      device gives us 67% occupancy.
 */
