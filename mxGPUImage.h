 
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
        float getValue(const int i, const int j);
        
        //Operators
        float* mxGPUImage::operator()(int , int);
        
        
        //Misc.
        mxGPUArray* getRegionAroundPixel(const int radius, 
                                    const int i, 
                                    const int j);
                                    
    private:
        int M,N;
        mxGPUArray* d_imagedata;
        float* d_imagedataPtr;
        bool image_data_exists;
        //static mwSize ndims;
};

//mwSize mxGPUImage::ndims=(mwSize) 2;

mxGPUImage::mxGPUImage(void):M(0),N(0),image_data_exists(false){
};

mxGPUImage::~mxGPUImage(void){
    if(image_data_exists==true){
        mxGPUDestroyGPUArray(d_imagedata);
    }
};

void mxGPUImage::setDataF(const mxGPUArray* A){
    const mwSize* array_Size=mxGPUGetDimensions(A);
    this->M=array_Size[0];
    this->N=array_Size[2];
    this->d_imagedata = mxGPUCopyGPUArray(A);
    this->d_imagedataPtr=(float*)mxGPUGetData(this->d_imagedata);
    image_data_exists=true; 
};

void mxGPUImage::setDataF(mxGPUArray* A){
    const mwSize* array_Size=mxGPUGetDimensions(A);
    this->M=array_Size[0];
    this->N=array_Size[2];
    this->d_imagedata =mxGPUCopyGPUArray(A);
    this->d_imagedataPtr=(float*) mxGPUGetData(this->d_imagedata);
    image_data_exists=true; 
};

float* mxGPUImage::getDataF(){
    if(image_data_exists==true)
        return this->d_imagedataPtr;
    else
        return NULL;
};

mxArray* mxGPUImage::getDataMxArray(){   
    if(image_data_exists==true){
        mxArray* h_imagedata=mxGPUCreateMxArrayOnCPU(this->d_imagedata);
        return h_imagedata;
    }
    else
        return NULL;     
};

mxGPUArray* mxGPUImage::getRegionAroundPixel(const int radius, 
                                        const int i, 
                                        const int j){
    
    mwSize* array_size=(mwSize*) mxMalloc((mwSize) 2*sizeof(mwSize));
    array_size[0]=2*radius+1;
    array_size[2]=array_size[0];

    mxGPUArray* d_region=mxGPUCreateGPUArray(   (mwSize) 2,
                                                array_size,
                                                mxSINGLE_CLASS,
                                                mxREAL,
                                                MX_GPU_DO_NOT_INITIALIZE);
    float* d_regiondata=(float*)mxGPUGetData(d_region);
    
    const int min_i=i-radius;
    const int max_i=i+radius;
    const int min_j=j-radius;
    const int max_j=j+radius;
    
    for (int i=min_i; i <= max_i; i++){
        for(int j=min_j;j <= max_j; j++){
            d_regiondata[i*(array_size[0]+j)]=this->getValue(i,j);
        }
    }

    return d_region;
};

float mxGPUImage::getValue(const int i, const int j){
    return *((this->d_imagedataPtr)+(j*(this->M)+i));
};

//Not sure if I want a pointer or a value
float* mxGPUImage::operator()(int i, int j){
    if (i < M && j < N)
        return ((this->getDataF())+(j*(this->M)+i));
    else
        return NULL;
};


#endif