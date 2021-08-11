#include <iostream>
#include <fstream>
#include <istream>
#include <vector>

#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/StdVector"

void mnistToMatrix(Eigen::MatrixXf& X, Eigen::MatrixXf& Y, const char* imageSetLocation, const char* labelLocation) {
    std::ifstream images(imageSetLocation, std::ios::in | std::ios::binary);
    
    uint32_t amtImgs;
    
    images.seekg(sizeof(uint32_t));
    images.read(reinterpret_cast<char*>(&amtImgs), sizeof(uint32_t));
    amtImgs = __builtin_bswap32(amtImgs);

    uint32_t vRes,hRes;
    
    images.seekg(2*sizeof(uint32_t));
    images.read(reinterpret_cast<char*>(&vRes), sizeof(uint32_t));
    vRes = __builtin_bswap32(vRes);

    images.seekg(3*sizeof(uint32_t));
    images.read(reinterpret_cast<char*>(&hRes), sizeof(uint32_t));
    hRes = __builtin_bswap32(hRes);

    std::cout << "amtImgs = "  << amtImgs << std::endl;
    std::cout << "vRes, hRes = " << vRes << ", " << hRes << std::endl;

    int exampleSize = vRes*hRes;
    int amtPix = amtImgs*exampleSize;

    unsigned char* cBuf = (unsigned char*)malloc(amtPix);
    float* fBuf = (float*)malloc(amtPix*sizeof(float));
    
    for(int i = 0 ; i < amtImgs; i++) {
        images.seekg(4*sizeof(uint32_t) + i*(exampleSize));
        images.read((char*)&cBuf[i*exampleSize], exampleSize);
    }

    for(int i = 0 ; i < amtPix ; i++) {
        fBuf[i] = cBuf[i];
    }

    Eigen::Map<Eigen::MatrixXf> X1(fBuf, exampleSize, amtImgs);
    X.resize(exampleSize, amtImgs);
    X = X1.transpose(); // Avoid going out of scope
    
    std::free(fBuf);
    std::free(cBuf);

    std::ifstream labels(labelLocation, std::ios::in | std::ios::binary);

    // Read amt labels
    int amtLabels;
    labels.seekg(sizeof(uint32_t));
    labels.read(reinterpret_cast<char*>(&amtLabels), sizeof(uint32_t));
    amtLabels = __builtin_bswap32(amtLabels); // Flip endianness
    std::cout << "amtLabels = " << amtLabels << std::endl;

    unsigned char* cBufL = (unsigned char*)malloc(amtLabels);
    Y.resize(amtLabels,10);
    Y.setZero();
    for(int i = 0 ; i < amtLabels ; i++) {
        Y(i,(int)cBufL[i]) = 1.0;
    }

    labels.seekg(2*sizeof(uint32_t));
    labels.read(reinterpret_cast<char*>(&amtLabels), sizeof(uint32_t));
}
