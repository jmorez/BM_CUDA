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

void __device__ lin_interp(     const double* image, 
                                double m, 
                                double n, 
                                const int imagerows,
                                double& interpolated){
    int k=(int) n*imagerows+m;
    interpolated=image[k];
}

//Check the type, I think they should be double const * const
void __global__ findMatches(double* const d_similarity,
                            const double* d_Cx,
                            const double* d_Cy,
                            const double* d_ref,
                            const int blocksize,
                            const double* d_searchwindow, //Note that it should include padding with (blocksize-1)/2
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
        //We must go deeper...
        int k; int m_r; int n_r;
        double x; double y; double x_r; double y_r;
        for (int m=0; m < blocksize*blocksize; m++){
        for (int n=0; n < blocksize; n++){
            k=blocksize*n+m;
            if(d_mask[k]==1){
                /* Calculate corresponding normalized coordinates
                 */
                x=n/(blocksize-1)-0.5;
                y=-m/(blocksize-1)+0.5;
                
                /* Check this expression! Notice that there's actually only
                 * 2 values in the parentheses. 
                 */
                /* Rotate coordinates (get everything working without 
                 * rotation first!).
                 */
                x_r=(Cx_r*Cx_m+Cy_r*Cy_m)*x+(Cx_m*Cy_r-Cx_r*Cy_m)*y;
                y_r=(Cx_r*Cy_m-Cx_m*Cy_r)*x+(Cx_r*Cx_m+Cy_r*Cy_m)*y;
                
                /* Return to indices, but immediately offset them so they 
                 * correspond to searchwindow coordinates
                 */
                m_r=(x_r+0.5)*((double)(blocksize-1))+(double)i;
                n_r=(0.5-y_r)*((double)(blocksize-1))+(double)j;
                double interpolated=0;
                lin_interp(d_searchwindow,m_r,n_r,window_size[0],interpolated);
                double d=d_ref[k]-interpolated;
                d_similarity[j*window_size[0]+i]=d*d;

                /*Transform to a linear index that can fetch the potential
                 *match pixel within the search window, interpolating. 
                 */
                
            }
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
    double* d_similarity;
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
    d_similarity = (double* const)(mxGPUGetData(similarity));
            
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
