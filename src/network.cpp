#include <iostream>
#include <vector>
#include <cmath>

#include "eigen3/Eigen/Core"

#include "nmath.h"
#include "network.h"
#include "utils.h"

void NeuralNetwork::gradientChecking() {
    static float epsilon = 2;
    static std::vector<float> numGrads(this->weights.size()-1);
    static std::vector<Eigen::MatrixXf> theta;

    for(std::vector<Eigen::MatrixXf*>::iterator it = this->weights.begin() ; it != this->weights.end() ; it++) {
        theta[it - this->weights.begin()] = **it;
    }
}

void NeuralNetwork::checkAll() {
    std::cout << "Testing all training inputs\n";

    for(int i = 0 ; i < m ; i++) {
        std::cout << "Training example i=" << this->m << std::endl;
        *this->layers[0] = this->X->col(i);
        this->forwardProp();
        this->readLayer(0);
        this->readLayer(this->L-1);
        std::cout << std::endl;
    }
}

void NeuralNetwork::readLayer(int l) {
    std::cout << "Layer " << l << ":\n" << *this->layers[l] << std::endl;
}

void NeuralNetwork::readOutputs() {
    std::cout << "Outputs:\n" << *this->layers[this->L-1] << std::endl;
}

float NeuralNetwork::cost() {
    /*
        Temporary Variables
    */
    static Eigen::ArrayXf out = Eigen::ArrayXf(this->K);
    static float J;
    
    J = 0;

    for(int i = 0 ; i < this->m ; i++) {
        *this->layers[0] = this->X->col(i);
        forwardProp();
        out = this->layers[this->L-1]->array();
        J += ((this->Y->col(i).array() * out.log())+(1.0 - this->Y->col(i).array()) * ((1.0 - out).log())).sum();
        //TODO: find a fix for perfect predictions (i.e. log(0))
    }

    return J*-this->mi;
}

float NeuralNetwork::cost(std::vector<Eigen::MatrixXf>& theta) {
    
    /*
        Temporary Variables
    */
    static Eigen::ArrayXf out = Eigen::ArrayXf(this->K);
    static float J;
    J = 0;

    this->forwardProp(theta);

    for(int i = 0 ; i < this->m ; i++) {
        out = this->layers[this->L-1]->array();

        J += ((this->Y->row(i).array() * out.log()) + (1 - this->Y->row(i).array())*(1 - out).log()).sum()/(-this->m);
    }

    return J;
}

int NeuralNetwork::gradientDescent() {
    this->forwardProp();
    this->backProp();
    static float max,prevmax = 0;
    max = 0;
    for(int l = 0 ; l < L-1 ; l++) {
        *this->weights[l] -= alpha * *gradients[l];
        if(gradients[l]->maxCoeff() > max) {
            max = gradients[l]->maxCoeff();
        }
    }
    prevmax = max;
    
    return 0;
}

int NeuralNetwork::backProp() {
    for(int i = 1 ; i < this->L-1 ; i++) {
        this->delta[i]->setZero();
    }

    for(int i = 0 ; i < this->m ; i++){
        *this->layers[0] = this->X->col(i);
        this->forwardProp();

        *this->costs[L-1] = *this->layers[L-1] - this->Y->col(i);
        *this->costs[L-2] = ((this->weights[this->L-2]->transpose() * *this->costs[this->L-1]).array() * this->midBLayers[this->L-2]->unaryExpr(&NNM::sigmoidp).array()).matrix();
        for(int k = L-3 ; k > 0 ; k--) {
            *this->costs[k] = ((this->weights[k]->transpose() * costs[k+1]->bottomRows(this->layerSizes[k+1])).array() * this->midBLayers[k]->unaryExpr(&NNM::sigmoidp).array()).matrix();
        }
        for(int k = 0 ; k < L-1 ; k++) {
            *this->delta[k] += this->costs[k+1]->bottomRows(this->layerSizes[k+1]) * this->bLayers[k]->transpose();
            *gradients[k] = *delta[k]*this->mi;
        }

        for(int k = 0 ; k < this->L-2 ; k++) {
                        
        }
    }
    return 0;
}

int NeuralNetwork::forwardProp() { // Performs forward propagation on the network, updating
    for(int currentLayer = 0 ; currentLayer < L-1 ; currentLayer++) {
        this->bLayers[currentLayer]->bottomRows(this->layerSizes[currentLayer]) = *this->layers[currentLayer];
        this->midBLayers[currentLayer+1]->bottomRows(this->layerSizes[currentLayer+1]) = (*this->weights[currentLayer] * *this->bLayers[currentLayer]);
        *this->layers[currentLayer+1] = this->midBLayers[currentLayer+1]->block(1,0,this->layerSizes[currentLayer+1],1).unaryExpr(&NNM::sigmoid);
        
        //this->bLayers[currentLayer+1]->block(0,1,1,this->layerSizes[currentLayer+1]) = *this->layers[currentLayer+1];
    }

    return 0;
}

int NeuralNetwork::forwardProp(std::vector<Eigen::MatrixXf>& theta) { // Performs forward propagation on the network, updating
    for(int currentLayer = 0 ; currentLayer < L-1 ; currentLayer++) {
        *this->midLayers[currentLayer+1] = ((1,*this->layers[currentLayer]) * theta[currentLayer].transpose());
        *this->layers[currentLayer+1] = this->midLayers[currentLayer+1]->unaryExpr(&NNM::sigmoid);
    }

    return 0;
}

int NeuralNetwork::setTraining(Eigen::MatrixXf* inputs, Eigen::MatrixXf* outputs) {
    this->X = inputs;
    this->Y = outputs;

    this->m = this->X->cols();
    this->n = this->X->rows();
    this->mi = 1.0/(float)m;
    return 0;
}

NeuralNetwork::NeuralNetwork(std::vector<int>& sl)
{
    this->layerSizes = sl;

    this->L = layerSizes.size();
    this->K = layerSizes[L-1];
    
    this->weights.resize(L-1);
    this->gradients.resize(L-1);

    // The first element of the following two are redundant, but are retained for ease of calculation
    this->costs.resize(L);
    this->delta.resize(L-1);

    this->layers.resize(L);
    this->bLayers.resize(L);
    this->midLayers.resize(L);
    this->midBLayers.resize(L);

    for(int l = 0 ; l < this->L-1; l++) {
        this->weights[l] = new Eigen::MatrixXf(this->layerSizes[l+1], this->layerSizes[l]+1);
        this->gradients[l] = new Eigen::MatrixXf(this->layerSizes[l+1], this->layerSizes[l]+1);

        this->delta[l] = new Eigen::MatrixXf(this->layerSizes[l+1], this->layerSizes[l]+1);
        this->costs[l+1] = new Eigen::MatrixXf(this->layerSizes[l+1]+1,1);

        // Randomise weights to break symmetry
        this->weights[l]->setRandom();
    }

    this->costs[this->L-1]->resize(1,this->layerSizes[this->L-1]);

    for(int l = 0 ; l < this->L; l++) {
        this->layers[l] = new Eigen::MatrixXf(layerSizes[l],1);
        this->bLayers[l] = new Eigen::MatrixXf(layerSizes[l]+1,1);
        (*this->bLayers[l])(0) = 1;
        this->midLayers[l] = new Eigen::MatrixXf(layerSizes[l],1);
        this->midBLayers[l] = new Eigen::MatrixXf(layerSizes[l]+1,1);
    }
}

NeuralNetwork::~NeuralNetwork() {
    for(std::vector<Eigen::MatrixXf*>::iterator it = this->weights.begin() ; it != this->weights.end() ; it++) {
        (**it).resize(0,0);
    }
    for(std::vector<Eigen::MatrixXf*>::iterator it = this->gradients.begin() ; it != this->gradients.end() ; it++) {
        (**it).resize(0,0);
    }
    for(std::vector<Eigen::MatrixXf*>::iterator it = this->delta.begin() ; it != this->delta.end() ; it++) {
        (**it).resize(0,0);
    }
    for(std::vector<Eigen::MatrixXf*>::iterator it = this->layers.begin() ; it != this->layers.end() ; it++) {
        (**it).resize(0,0);
    }
    for(std::vector<Eigen::MatrixXf*>::iterator it = this->bLayers.begin() ; it != this->bLayers.end() ; it++) {
        (**it).resize(0,0);
    }
    for(std::vector<Eigen::MatrixXf*>::iterator it = this->midLayers.begin() ; it != this->midLayers.end() ; it++) {
        (**it).resize(0,0);
    }
    for(std::vector<Eigen::MatrixXf*>::iterator it = this->midBLayers.begin() ; it != this->midBLayers.end() ; it++) {
        (**it).resize(0,0);
    }
    for(std::vector<Eigen::MatrixXf*>::iterator it = this->costs.begin()+1 ; it != this->costs.end() ; it++) {
        (**it).resize(0,0);
    }
}