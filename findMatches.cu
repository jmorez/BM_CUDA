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

/* TO DO:
 *verify every input:
 *everything should be single-precision 
 *Cx and Cy should match the size of searchwindow and should have odd size
 *ref must have M=N (odd)
 *mask should be bool and the same size as ref
*/
// Texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> tex;

//Check the type (cfr. mexGPUExample.cu), I think they should be float const * const
void __global__ findMatches(float* const  d_similarity,
                            const float* const d_Cx,
                            const float* const d_Cy,
                            const float* const d_ref, // <-- THIS SHOULD ALSO BE A TEXTURE
                            const int blocksize,
                            //const float* const d_searchwindow, //Note that it should include padding. 
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

        int k;
        float x; float y;
        float R11; float R12;
        float x_r; float y_r;
        float m_r; float n_r;
        float u; float v;
        //float d=(float)0;
        float d;

        for (int n=0; n < blocksize; n++){
        for (int m=0; m < blocksize; m++){
            k=blocksize*n+m; //corresponding reference linear index
            if(d_mask[k]==true){
                
                //Rotation coordinates
                x=(float)n/((float)blocksize-1.)-0.5;
                y=-(float)m/((float)blocksize-1.)+0.5;
             
                //Rotation matrix
                R11=(Cx_r*Cx_m+Cy_r*Cy_m);
                R12=(Cx_m*Cy_r-Cx_r*Cy_m);
                
                //Rotate coordinates
                x_r=R11*x-R12*y;
                y_r=R12*x+R11*y;
                
                //Transform back to array coordinates of the potential match.
                m_r=(0.5-(float)y_r)*((float)blocksize-1.);
                n_r=((float)x_r-0.5)*((float)blocksize-1.);
                
                //Transform to coordinates within the search window
                u=(float)(m_r+(float)i-(float)padding_size);
                v=(float)(n_r+(float)j-(float)padding_size);
                
                //Calculate difference
                d=d_ref[k]-tex2D(tex,u,v);
                //Square it
                d_similarity[j*window_M+i]+=d*d;
            }
        }
        }
    }
        else if(i < window_M && j < window_N)
            d_similarity[j*window_M+i]=(float)0;        
}

void mexFunction(   int nlhs, mxArray *plhs[],
                    int nrhs, mxArray const *prhs[]){
    
 /* ########################Input Verification############################
  * If you're fearless, you comment this out and gain a couple of milliseconds.
  */
    
    
    char const * const errId            = "parallel:gpu";
    char const * const err_Arguments    = "Incorrect amount of arguments.";
    char const * const err_Type         = "First four arguments must be of single precision type.";
    char const * const err_TypeLogical  = "Mask must be of type logical.";
    char const * const err_RefWrongSize = "Reference size should be odd.";
    char const * const err_SWWrongSize  = "Searchwindow size should be odd.";
    char const * const err_CWrongSize   = "Cx and Cy should be the same size as the searchwindow";
    
    /* Check the amount of arguments */
    if(nrhs!=5)
        mexErrMsgIdAndTxt(errId, err_Arguments);
    
    /*Check the types */
    if(mxIsSingle(prhs[0])==false 
      |mxIsSingle(prhs[1])==false
      |mxIsSingle(prhs[2])==false
      |mxIsSingle(prhs[3])==false)
        mexErrMsgIdAndTxt(errId, err_Type);
    
    if(mxIsLogical(prhs[4])==false)
        mexErrMsgIdAndTxt(errId, err_TypeLogical);
    
    /*Check the size of the reference block */
    if(mxGetM(prhs[2]) % 2 ==0 | mxGetN(prhs[2]) % 2 == 0)
        mexErrMsgIdAndTxt(errId, err_RefWrongSize);
    /*Check the size of the search window */
    if(mxGetM(prhs[3]) % 2 ==0 | mxGetN(prhs[3]) % 2 == 0)
        mexErrMsgIdAndTxt(errId, err_SWWrongSize);
    
    /* Make sure Cx and Cy are the same size as searchwindow */
    if(mxGetM(prhs[3])!=mxGetM(prhs[0])
      |mxGetM(prhs[3])!=mxGetM(prhs[1])
      |mxGetN(prhs[3])!=mxGetN(prhs[0])     
      |mxGetN(prhs[3])!=mxGetN(prhs[1]))
        mexErrMsgIdAndTxt(errId, err_CWrongSize);
      
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

   
    //Create the image array on the GPU and assign device pointers.
    Cx              =mxGPUCreateFromMxArray(prhs[0]);
    Cy              =mxGPUCreateFromMxArray(prhs[1]);
    ref             =mxGPUCreateFromMxArray(prhs[2]);
    searchwindow    =mxGPUCreateFromMxArray(prhs[3]);
    mask            =mxGPUCreateFromMxArray(prhs[4]);
    
    const float* const d_Cx    =(float* const)mxGPUGetDataReadOnly(Cx);
    const float* const d_Cy    =(float* const)mxGPUGetDataReadOnly(Cy);
    const float* const d_ref   =(float* const)mxGPUGetDataReadOnly(ref);
    const float* const d_searchwindow =(float* const)mxGPUGetDataReadOnly(searchwindow);  
    const bool*  const d_mask  =(bool*  const)mxGPUGetDataReadOnly(mask);
    
    
    
    
    //Output array creation
    similarity = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(searchwindow),
                            mxGPUGetDimensions(searchwindow),
                            mxSINGLE_CLASS,
                            mxREAL,
                            MX_GPU_INITIALIZE_VALUES); //Initialize to 0
    
    float* const d_similarity = (float*)mxGPUGetData(similarity);
          
    
    //Get blocksize and windowsize including padding
    const mwSize* mw_blocksize=mxGPUGetDimensions(ref);
    const int blocksize=mw_blocksize[0];

    const mwSize* mw_WindowSize =  mxGPUGetDimensions(searchwindow);
    const int window_M=mw_WindowSize[0];
    const int window_N=mw_WindowSize[1];    

    
    /* Assign d_searchwindow to a texture for performance purposes */

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
    //For 2D textures this is actually bilinear filtering
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = false;    

    // Bind the array to the texture
    cudaBindTextureToArray(tex, cuArray, channelDesc);
     
    
    
    /*Kernel parameters
	 *Figure out grid layout. Assuming device of compute capability 2.x or more */
	cudaDeviceProp device;
	cudaGetDeviceProperties(&device,0);
	const int MaxThreadsPerBlock=device.maxThreadsPerBlock;

	threadsPerBlock.x=(size_t)sqrt((float) MaxThreadsPerBlock);
	threadsPerBlock.y=(size_t)sqrt((float) MaxThreadsPerBlock);
   
    blocksPerGrid.x=(size_t)(window_M-1)/threadsPerBlock.x+1;
    blocksPerGrid.y=(size_t)(window_N-1)/threadsPerBlock.y+1;
    blocksPerGrid.z=1;
    
    //Run kernel
    findMatches<<<blocksPerGrid,threadsPerBlock>>>( d_similarity,
                                                    d_Cx,
                                                    d_Cy,
                                                    d_ref,
                                                    blocksize,
                                                    //d_searchwindow,
                                                    window_M,
                                                    window_N,
                                                    d_mask);
    cudaDeviceSynchronize();
    
    //Return output
    plhs[0] = mxGPUCreateMxArrayOnCPU(similarity);
    
    //Garbage collection
    cudaUnbindTexture(tex); 	
    cudaFreeArray(cuArray);
    mxGPUDestroyGPUArray(Cx);
    mxGPUDestroyGPUArray(Cy);
    mxGPUDestroyGPUArray(ref);
    mxGPUDestroyGPUArray(searchwindow);
    mxGPUDestroyGPUArray(mask);
    mxGPUDestroyGPUArray(similarity);
}

