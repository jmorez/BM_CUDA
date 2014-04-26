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
void __global__ findMatches(double* const  d_similarity,
                            const double* const d_Cx,
                            const double* const d_Cy,
                            const double* const d_ref,
                            const int blocksize,
                            //const double* const d_searchwindow, //Note that it should include padding. 
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
    double Cx_r=d_Cx[center];
    double Cy_r=d_Cy[center];


    const int pm_centroid=j*window_M+i;
    double Cx_m=d_Cx[pm_centroid];
    double Cy_m=d_Cy[pm_centroid];
    const int padding_size=(int)(blocksize-1)/2;          
        if (    i < window_M-padding_size   && j < window_N-padding_size 
             && i > padding_size            && j > padding_size){
        //d_similarity[j*window_M+i]=(float)tex2D(tex,(float)j/(float)window_N,(float)i/(float)window_M);   
        int k;
        double x; double y;
        double R11; double R12;
        double x_r; double y_r;
        double m_r; double n_r;
        float u; float v;
        double d=(double)0;
        
        for (int n=0; n < blocksize; n++){
        for (int m=0; m < blocksize; m++){
            k=blocksize*n+m; //corresponding reference coordinate
            if(d_mask[k]==true){
                x=n/(blocksize-1)-0.5;
                y=-m/(blocksize-1)+0.5;
             

                R11=(Cx_r*Cx_m+Cy_r*Cy_m);
                R12=(Cx_m*Cy_r-Cx_r*Cy_m);
                
                x_r=R11*x-R12*y;
                y_r=R12*x+R11*y;
                
                m_r=(0.5-y_r)*(blocksize-1);
                n_r=(x_r-0.5)*(blocksize-1);
                u=(float)((m_r+(float)i)/(float)window_M);
                v=(float)((n_r+(float)j)/(float)window_N);
                
                d=d_ref[k]-(double)tex2D(tex,u,v);
                d_similarity[j*window_M+i]+=(double)1;//d*d;//tex2D(tex,(float)i/(float)window_M,(float)j/(float)window_N);    
            }
        }
        }
    }
        else if(i < window_M && j < window_N)
            d_similarity[j*window_M+i]=(double)0;        
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
    
    const double* const  d_Cx    =(double* const)mxGPUGetDataReadOnly(Cx);
    const double* const  d_Cy    =(double* const)mxGPUGetDataReadOnly(Cy);
    const double* const  d_ref   =(double* const)mxGPUGetDataReadOnly(ref);
    const bool*  const  d_mask  =(bool* const)mxGPUGetDataReadOnly(mask);
    
    const float* const  d_searchwindow =(float* const)mxGPUGetDataReadOnly(searchwindow);  
    
    
    //Output array, probably needs a smaller size due to padding of searchwindow
    similarity = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(searchwindow),
                            mxGPUGetDimensions(searchwindow),
                            mxDOUBLE_CLASS,
                            mxGPUGetComplexity(searchwindow),
                            MX_GPU_DO_NOT_INITIALIZE);
    double* const d_similarity = (double*)mxGPUGetData(similarity);
          
    
    
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
                      (const void*)d_searchwindow,
                      size,
                      cudaMemcpyDeviceToDevice);

    // Set texture parameters
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = true;    

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
                                                    //d_searchwindow,
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
