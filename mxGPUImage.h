#include "gpu/mxGPUArray.h" 

#ifndef MXGPUIMAGE_H
#define MXGPUIMAGE_H
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
#endif