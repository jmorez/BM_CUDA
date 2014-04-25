/*  ###########################DESCRIPTION###############################
 *  Written by XXXXXXX (University of XXXXXX) as a part of a bachelor's 
 *  thesis that uses a blockmatching algorithm  to gather a statistical 
 *  population for denoising single pixels in an image.
 */

 /* ###########################STYLE NOTES###############################
  * Device variables will have the prefix d_, no prefix implies a host 
  * variable.
  */

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <math.h>
#include <cuda_runtime.h>

/*  This kernel will find matches for a reference block and a search window. 
 *  Every thread gets one comparison of the reference block with a block 
 *  from the search window. 
 *  The search window is assumed to include padding (either with zeroes or 
 *  an extra part of the image. 
 *  The centroid components Cx and Cy are assumed to be normalized.
 *  
 */

// Texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> tex;

//Check the type (cfr. mexGPUExample.cu), I think they should be float const * const
void __global__ findMatches(float* const  d_similarity,
                            const float* const d_Cx,
                            const float* const d_Cy,
                            const float* const d_ref,
                            const int blocksize,
                            const float* const d_searchwindow, //Note that it should include padding. 
                            const int window_M,
                            const int window_N, 
                            const bool* const d_mask){
                            
    /*  Coordinates of the center of a potential match, accounting for padding 
     *  of the search window.
     */
    
    const int i = blockDim.x*blockIdx.x+threadIdx.x;
    const int j = blockDim.y*blockIdx.y+threadIdx.y;
    /*  Fetch the reference and match centroid components, I might pass this 
     *  to the kernel because this is the same for every thread.
     */

    const int center=(int)((window_M)*(window_N)-1)/2;
    float Cx_r=d_Cx[center];
    float Cy_r=d_Cy[center];


    const int pm_centroid=j*window_M+i;
    float Cx_m=d_Cx[pm_centroid];
    float Cy_m=d_Cy[pm_centroid];
    const int padding_size=(int)(blocksize-1)/2;          
        if (    i < window_M-padding_size   && j < window_N-padding_size 
             && i > padding_size            && j > padding_size){
           d_similarity[j*window_M+i]=tex2D(tex,(float)i/(float)window_M,(float)j/(float)window_N);
//         int k; int m_r; int n_r; int u; int v; int k_m;
//         float x; float y; float x_r; float y_r;
//         for (int n=0; n < blocksize; n++){
//         for (int m=0; m < blocksize; m++){
//             k=blocksize*n+m;
//             u=i-padding_size+m;
//             v=j-padding_size+n;
//             k_m=v*window_M+u;
//             if(d_mask[k]==true){
//                 /* Calculate corresponding normalized coordinates
//                  */
//                 x=n/(blocksize-1)-0.5;
//                 y=-m/(blocksize-1)+0.5;
//                 
//                 /* Check this expression! Notice that there's actually only
//                  * 2 values in the parentheses. 
//                  */
//                 /* Rotate coordinates (get everything working without 
//                  * rotation first!).
//                  */
//                 /*
//                 x_r=(Cx_r*Cx_m+Cy_r*Cy_m)*x+(Cx_m*Cy_r-Cx_r*Cy_m)*y;
//                 y_r=(Cx_r*Cy_m-Cx_m*Cy_r)*x+(Cx_r*Cx_m+Cy_r*Cy_m)*y;
//                 */
//                 x_r=1*x+0*y;
//                 y_r=0*x+1*y;
//                 /* Return to indices, but immediately offset them so they 
//                  * correspond to searchwindow coordinates
//                  */
//                 m_r=(x_r+0.5)*((float)(blocksize-1))+(float)i;
//                 n_r=(0.5-y_r)*((float)(blocksize-1))+(float)j;
//                 /*
//                 float interpolated=0;
//                 lin_interp(d_searchwindow,m_r,n_r,window_size[0],interpolated);
//                 float d=5+0*d_ref[k]+0*interpolated;
//                  */
//                 float d=d_searchwindow[k_m]-d_ref[k];
//                 d_similarity[j*window_M+i]=1;//tex2D(tex, 0.5, 0.5);
// 
//                 /*Transform to a linear index that can fetch the potential
//                  *match pixel within the search window, interpolating. 
//                  */  
//             }
//         }
//         }
    }
        else if(i < window_M && j < window_N)
            d_similarity[j*window_M+i]=(float)0;        
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
     
    //mxGPUArray & device pointer declarations, make sure to destroy mxGPUArrays.
    mxGPUArray* similarity;
    const mxGPUArray* Cx; 
    //const float* d_Cx;
    const mxGPUArray* Cy;
    //const float* d_Cy; 
    const mxGPUArray* ref;
    //const float* d_ref;
    const mxGPUArray* searchwindow;
    //const float* d_searchwindow;
    const mxGPUArray* mask;
    //const bool* d_mask;
    
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
    
    const float* const  d_Cx    =(float* const)mxGPUGetDataReadOnly(Cx);
    const float* const  d_Cy    =(float* const)mxGPUGetDataReadOnly(Cy);
    const float* const  d_ref   =(float* const)mxGPUGetDataReadOnly(ref);
    const float* const  d_searchwindow =(float* const)mxGPUGetDataReadOnly(searchwindow);
    const bool*  const  d_mask  =(bool* const)mxGPUGetDataReadOnly(mask);
    
    
    
    
    //Output array, probably needs a smaller size due to padding of searchwindow
    similarity = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(searchwindow),
                            mxGPUGetDimensions(searchwindow),
                            mxSINGLE_CLASS,
                            mxGPUGetComplexity(searchwindow),
                            MX_GPU_DO_NOT_INITIALIZE);
    float* const d_similarity = (float*)mxGPUGetData(similarity);
          
    
    
    //Get blocksize and windowsize including padding
    const mwSize* mw_blocksize=mxGPUGetDimensions(ref);
    const int blocksize=mw_blocksize[0];

    const mwSize* mw_WindowSize =  mxGPUGetDimensions(searchwindow);
    const int window_M=mw_WindowSize[0];
    const int window_N=mw_WindowSize[2];

    //Read d_searchwindow into a texture
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    
    unsigned int size=window_M*window_N*sizeof(float);
    cudaArray *cuArray;
    cudaMallocArray(    &cuArray,
                        &channelDesc,
                        window_M,
                        window_N);
    cudaMemcpyToArray(cuArray,
                      0,
                      0,
                      d_searchwindow,
                      size,
                      cudaMemcpyHostToDevice);

    // Set texture parameters
    tex.addressMode[0] = cudaAddressModeWrap;
    tex.addressMode[1] = cudaAddressModeWrap;
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = true;    // access with normalized texture coordinates

    // Bind the array to the texture
    cudaBindTextureToArray(tex, cuArray, channelDesc);
     
    /*Kernel parameters
	 *Figure out grid layout. (1)
	 */
	cudaDeviceProp device;
	cudaGetDeviceProperties(&device,0);
	const int MaxThreadsPerBlock=device.maxThreadsPerBlock;

	threadsPerBlock.x=(size_t)sqrt((float) MaxThreadsPerBlock);
	threadsPerBlock.y=(size_t)sqrt((float) MaxThreadsPerBlock);
   
    blocksPerGrid.x=(size_t)(window_M-1)/threadsPerBlock.x+1;
    blocksPerGrid.y=(size_t)(window_N-1)/threadsPerBlock.y+1;
    blocksPerGrid.z=1;
    
    findMatches<<<blocksPerGrid,threadsPerBlock>>>( d_similarity,
                                                    d_Cx,
                                                    d_Cy,
                                                    d_ref,
                                                    blocksize,
                                                    d_searchwindow,
                                                    window_M,
                                                    window_N,
                                                    d_mask);
     
    plhs[0] = mxGPUCreateMxArrayOnCPU(similarity);
    
    mexPrintf("\n");
    
    cudaFreeArray(cuArray);
    mxGPUDestroyGPUArray(Cx);
    mxGPUDestroyGPUArray(Cy);
    mxGPUDestroyGPUArray(ref);
    mxGPUDestroyGPUArray(searchwindow);
    mxGPUDestroyGPUArray(mask);
    mxGPUDestroyGPUArray(similarity);
}

/*####################### FOOTNOTES ##################################
*/
