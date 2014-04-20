 
#ifndef MXGPUIMAGE_H
#define MXGPUIMAGE_H
#include "gpu/mxGPUArray.h"

/* Wrapper class with all kinds of useful methods for indexing and stuff 
 * like that.
 */
class mxGPUImage{
    public:
        mxGPUImage(void);
        ~mxGPUImage(void);
        
        //set-functions
        void setDataF(const mxGPUArray*);
        void setDataF(mxGPUArray*);
        void setDataF(float* );
        
        //get-functions
        float* getDataF(void);
        mxArray* getDataMxArray(void);
        int getM(void);
        int getN(void);
        
        //Operators
        float* mxGPUImage::operator()(int , int);
        
        
        //Misc.
        float* getRegionAroundPixel(const int radius, 
                                    const int i, 
                                    const int j);
                                    
    private:
        int M,N;
        float* d_imagedata;
        bool image_data_exists;
};

mxGPUImage::mxGPUImage(void):M(0),N(0),image_data_exists(false){
};

mxGPUImage::~mxGPUImage(void){
    if(image_data_exists==true){
        //free(d_imagedata);
    }
};

void mxGPUImage::setDataF(const mxGPUArray* A){
    const mwSize* array_Size=mxGPUGetDimensions(A);
    this->M=array_Size[0];
    this->N=array_Size[2];
    this->d_imagedata =(float*) mxGPUGetDataReadOnly(A);
    image_data_exists=true; 
};

void mxGPUImage::setDataF(mxGPUArray* A){
    const mwSize* array_Size=mxGPUGetDimensions(A);
    this->M=array_Size[0];
    this->N=array_Size[2];
    this->d_imagedata =(float*) mxGPUGetData(A);
    image_data_exists=true; 
};

float* getRegionAroundPixel(    const int radius, 
                                    const int i, 
                                    const int j){
    return NULL;
};

//Not sure if I want a pointer or a value
float* mxGPUImage::operator()(int i, int j){
    if (i < M && j < N){
        return (this->d_imagedata)+sizeof(float)*(j*(this->M)+i);}
    else{
        return NULL;
    }
};


#endif