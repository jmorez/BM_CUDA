 
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
        void setDataF(double* );
        
        //get-functions
        double* getDataF(void);
        mxArray* getDataMxArrayCPU(void);
        int getM(void);
        int getN(void);
        double getValue(const int i, const int j);
        
        //Operators
        double mxGPUImage::operator()(int , int);
        
        
        //Misc.
        mxGPUArray* getRegionAroundPixel(const int radius, 
                                    const int i, 
                                    const int j);
                                    
    private:
        int M,N;
        mxGPUArray* d_imagedata;
        double* d_imagedataPtr;
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
    this->d_imagedataPtr=(double*)mxGPUGetData(this->d_imagedata);
    image_data_exists=true; 
};

void mxGPUImage::setDataF(mxGPUArray* A){
    const mwSize* array_Size=mxGPUGetDimensions(A);
    this->M=array_Size[0];
    this->N=array_Size[2];
    this->d_imagedata =mxGPUCopyGPUArray(A);
    this->d_imagedataPtr=(double*) mxGPUGetData(this->d_imagedata);
    image_data_exists=true; 
};


//Returns a pointer to the raw doubleing point image data

double* mxGPUImage::getDataF(){
    if(image_data_exists==true)
        return this->d_imagedataPtr;
    else
        return NULL;
};


//Returns an mxArray pointer to the data copied to the CPU 
mxArray* mxGPUImage::getDataMxArrayCPU(){   
    if(image_data_exists==true){
        mxArray* h_imagedata=mxGPUCreateMxArrayOnCPU(this->d_imagedata);
        return h_imagedata;
    }
    else
        return NULL;     
};


//Returns an mxGPUArray that contains a square region of edge 2*radius+1 
//around a pixel (i,j). This probably only works when called from a kernel.
mxGPUArray* mxGPUImage::getRegionAroundPixel(const int radius, 
                                        const int i, 
                                        const int j){
    if(image_data_exists==true){
        mwSize* array_size=(mwSize*) mxMalloc((mwSize) 2*sizeof(mwSize));
        array_size[0]=2*radius+1;
        array_size[2]=array_size[0];
        array_size[1]=0;
        array_size[3]=0;

        mxGPUArray* d_region=mxGPUCreateGPUArray(   (mwSize) 2,
                                                    array_size,
                                                    mxDOUBLE_CLASS,
                                                    mxREAL,
                                                    MX_GPU_DO_NOT_INITIALIZE);

        double* d_regiondata=(double*)mxGPUGetData(d_region);

        const int min_i=i-radius;
        const int max_i=i+radius;
        const int min_j=j-radius;
        const int max_j=j+radius;

        int k=0;
        for(int j=min_j;j <= max_j; j++){
            for (int i=min_i; i <= max_i; i++){ 
                //This just doesn't work, am I assigning a host to a device 
                //variable or vice versa?
                d_regiondata[k]=(this->getValue)(0,0);
                k+=1;
            }
        }
        return d_region;
    }
    else
        return NULL;    
};


//Useful for assigning single pixels a certain value. Cfr. supra.
double mxGPUImage::getValue(const int i, const int j){
    return (this->d_imagedataPtr)[j*(this->M)+i];
};

//Not sure if I want a pointer or a value
double mxGPUImage::operator()(int i, int j){
    if (i < M && j < N)
        return (this->d_imagedataPtr)[j*(this->M)+i];
    else
        return NULL;
};

int mxGPUImage::getM(){
    return this->M;
};


#endif