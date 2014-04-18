#include "gpu/mxGPUArray.h" 

#ifndef MXGPUIMAGE_H
#define MXGPUIMAGE_H


/* Wrapper class with all kinds of useful methods for indexing and stuff 
 * like that.
 */
class mxGPUImage {
    public:
        void setDataF(const mxGPUArray* );
        float mxGPUImage::operator()(int , int);

    private:
        float* d_imagedata;
        int M,N;
};

void mxGPUImage::setDataF(const mxGPUArray* d_array){
    const mwSize* array_Size=mxGPUGetDimensions(d_array);
    this->M=array_Size[0];
    this->N=array_Size[2];
    //This still needs allocation and crap like that
    float* d_arrayptr=(float*)mxGPUGetDataReadOnly(d_array);
    for (int i=0; i < M*N-1; i++){
        d_imagedata[i]=d_arrayptr[i];
    }
};
/*
=======
class mxGPUImage {
    public:
        mxGPUImage(void);
        mxGPUImage createFromGPUArray(mxGPUArray*);
        float mxGPUImage::operator()(int i, int j){
            return 0.;
        };
        /*
        mxGPUImage getRegionAroundPixel(int, int);
        int getM();
        int getN();
    private:
        const int M, N;
         */
};

mxGPUImage::mxGPUImage(void){
    mexPrintf("\n fak off \n");
};

mxGPUImage createFromGPUArray(mxGPUArray* d_array){
    mxGPUImage meh;
    return meh;
};

*/
float mxGPUImage::operator()(int i, int j){
    return 0.;
};
=======

#endif