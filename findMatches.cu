/*  ###########################DESCRIPTION###############################
 *  Written by XXXXXXX (University of XXXXXX) as a part of a bachelor's 
 *  thesis that uses a blockmatching algorithm  to gather a statistical 
 *  population for denoising single pixels in an image.
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


/*  This kernel will find matches for a reference block and a search window. 
 *  Every thread gets one comparison of the reference block with a block 
 *  from the search window. 
 *  The search window is assumed to include padding (either with zeroes or 
 *  an extra part of the image. 
 *  The centroid components Cx and Cy are assumed to be normalized.
 *  
 */
void __global__ findMatches(const double* d_similarity,
                            const double* d_Cx,
                            const double* d_Cy,
                            const double* d_ref,
                            const int blocksize,
                            const double* d_search_window, //Note that it should include padding with (blocksize-1)/2
                            const int* window_size, 
                            const bool* d_mask){
    /*  Coordinates of the center of a potential match, accounting for padding 
     *  of the search window.
     */
    const int padding_size=(int)(blocksize-1)/2;
    const int i = blockDim.x*blockIdx.x+threadIdx.x+padding_size;
    const int j = blockDim.y*blockIdx.y+threadIdx.y+padding_size;
    /*  Fetch the reference and match centroid components, I might pass this 
     *  to the kernel because this is the same for every thread.
     */
    const int searchwindow_center=(int)(window_size[0]*window_size[1]-1)/2;
    double Cx_r=d_Cx[searchwindow_center];
    double Cy_r=d_Cy[searchwindow_center];
    
    const int pm_centroid=(j-padding_size)*(window_size[0]-blocksize)+i-padding_size;
    double Cx_m=d_Cx[pm_centroid];
    double Cy_m=d_Cy[pm_centroid];
    
    if (    window_size[0] <= i && i < window_size[0] 
            &&window_size[1] <= j && j < window_size[1]){
        int m; int n;
        for (int k=0; k < blocksize*blocksize; k++){
            if(d_mask[k]==1){
                /*Calculate indices m and n that correspond with a pixel 
                 * within the reference block.
                 */
                
                /* Calculate corresponding normalized coordinates
                 */
                
                /* Rotate coordinates (get everything working without 
                 * rotation first!).
                 */
                
                /*Find rotated (non integer) indices
                 */
                
                /*Transform to a linear index that can fetch the potential
                 *match pixel within the search window, interpolating. 
                 */
                
            }
        }
    }
}

/* Call in matlab like this:
[plhs[0],plhs[1],plhs[...],plhs[nrhs-1]]=filename(prhs[0],prhs[1],prhs[...],prhs[nrhs-1])
*/
void mexFunction(   int nlhs, mxArray *plhs[],
                    int nrhs, mxArray const *prhs[]){
    /* prhs argument explanation:
     *plhs[0]: Cx
     *... [1]: Cy
     *... [2]: ref
     *... [3]: search_window
     *... [4]: mask
     */
    
    //Initialize MathWorks GPU API. 
    mxInitGPU();
    
    //Misc. array declarations
    int* window_size;
    
    
    //Array & device pointer declarations, make sure to destroy mxGPUArrays.
    mxGPUArray* similarity;
    const double* d_similarity;
    const mxGPUArray* Cx; 
    const double* d_Cx;
    const mxGPUArray* Cy;
    const double* d_Cy; 
    const mxGPUArray* ref;
    const double* d_ref;
    const mxGPUArray* searchwindow;
    const double* d_searchwindow;
    const mxGPUArray* mask;
    const bool* d_mask;
    
    //Grid parameters
    dim3 blocksPerGrid;
    dim3 threadsPerBlock;

    /*Input verification
        ... I'll do it later
    */
    //Create the image array on the GPU and assign device pointers.
    Cx              =mxGPUCreateFromMxArray(prhs[0]);
    Cy              =mxGPUCreateFromMxArray(prhs[1]);
    ref             =mxGPUCreateFromMxArray(prhs[2]);
    searchwindow    =mxGPUCreateFromMxArray(prhs[3]);
    mask            =mxGPUCreateFromMxArray(prhs[4]);
    
    d_Cx=(double*)mxGPUGetDataReadOnly(Cx);
    d_Cy=(double*)mxGPUGetDataReadOnly(Cy);
    d_ref=(double*)mxGPUGetDataReadOnly(ref);
    d_searchwindow=(double*)mxGPUGetDataReadOnly(searchwindow);
    d_mask=(bool*)mxGPUGetDataReadOnly(mask);
    
    //Output array, probably needs a smaller size due to padding of searchwindow
    similarity = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(searchwindow),
                            mxGPUGetDimensions(searchwindow),
                            mxGPUGetClassID(searchwindow),
                            mxGPUGetComplexity(searchwindow),
                            MX_GPU_DO_NOT_INITIALIZE);
    d_similarity = (double *)(mxGPUGetData(similarity));
            
    //Get blocksize and windowsize without padding
    const mwSize* mw_blocksize=mxGPUGetDimensions(ref);
    const int blocksize=mw_blocksize[0];

    const mwSize* mw_WindowSize =  mxGPUGetDimensions(searchwindow);
    window_size=(int*) mxMalloc(sizeof(int)*2);
    //Account for padding
    window_size[0]=mw_WindowSize[0]-blocksize;
    window_size[1]=mw_WindowSize[2]-blocksize;
    

     
    /*Kernel parameters
	 *Figure out grid layout. (1)
	 */
	cudaDeviceProp device;
	cudaGetDeviceProperties(&device,0);
	const int MaxThreadsPerBlock=device.maxThreadsPerBlock;

	threadsPerBlock.x=(size_t)sqrt((double) MaxThreadsPerBlock);
	threadsPerBlock.y=(size_t)sqrt((double) MaxThreadsPerBlock);
   
    blocksPerGrid.x=(size_t)(window_size[0]-1)/threadsPerBlock.x+1;
    blocksPerGrid.y=(size_t)(window_size[2]-1)/threadsPerBlock.y+1;
    blocksPerGrid.z=1;
    
    findMatches<<<blocksPerGrid,threadsPerBlock>>>( d_similarity,
                                                    d_Cx,
                                                    d_Cy,
                                                    d_ref,
                                                    blocksize,
                                                    d_searchwindow,
                                                    window_size,
                                                    d_mask);

   /*
    mexPrintf("\n");
    for(int i=0; i < 3; i++){
        mexPrintf("\n");
        for(int j=0; j < 3; j++){
        mexPrintf(" %f",output2Ptr[j*3+i]);
        }
    }
     */
    mexPrintf("\n");
    
    mxFree(window_size);
    mxGPUDestroyGPUArray(Cx);
    mxGPUDestroyGPUArray(Cy);
    mxGPUDestroyGPUArray(ref);
    mxGPUDestroyGPUArray(searchwindow);
    mxGPUDestroyGPUArray(mask);
}

/*####################### FOOTNOTES ##################################
*/
