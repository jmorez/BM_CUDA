#include "gpu/mxGPUArray.h" 

#ifndef MXGPUIMAGE_H
#define MXGPUIMAGE_H


/* Wrapper class with all kinds of useful methods for indexing and stuff 
 * like that.
 */
class mxGPUImage{
    public:
        int M,N;
        mxGPUImage(void);
        ~mxGPUImage(void);
        
        void setDataF(const mxGPUArray* );
        void printPixelValue(int i, int j);
        mxGPUImage getRegionAroundPixel(const int radius, 
                                    const int i, 
                                    const int j);
                                    
        float mxGPUImage::operator()(int , int); 
        
    private:
        float* d_imagedata;
        bool image_data_exists;
};

mxGPUImage::mxGPUImage(void):M(0),N(0),image_data_exists(false){
};

mxGPUImage::~mxGPUImage(void){
    if(image_data_exists==true){
        //mxGPUDestroyGPUArray(d_imagedata);
    }
}

void mxGPUImage::setDataF(const mxGPUArray* d_array){
    const mwSize* array_Size=mxGPUGetDimensions(d_array);
    this->M=array_Size[0];
    this->N=array_Size[2];
    //d_B = (double *)(mxGPUGetData(B));
    this->d_imagedata =(float*) mxGPUGetDataReadOnly(d_array);
    image_data_exists=true; 
};

void mxGPUImage::printPixelValue(int i, int j){
    //mxArray h_value mxGPUCreateMxArrayOnCPU();
    //float h_value=5.;
    mexPrintf("\n %f \n",(*this)(i,j));
};

mxGPUImage getRegionAroundPixel(    const int radius, 
                                    const int i, 
                                    const int j){
    mxGPUImage nothing;
    return nothing;
};

float mxGPUImage::operator()(int i, int j){
    //I so hope this works
    return (this->d_imagedata)[j*(this->M)+i];
    //return  ((float*)mxGPUGetData(d_imagedata))[j*(this->M)+i];
};


#endif