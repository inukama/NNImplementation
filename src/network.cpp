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
    std::cout << "Printing all layers\n";
    for(int l = 0 ; l < L ; l++) {
        for(int i = 0 ; i < this->layerSizes[l] ; i++) {
            std::cout << layers[l] << "\n";
        }
        std::cout << "\n";
    }

    std::cout << "Testing all training inputs\n";

    for(int i = 0 ; i < m ; i++) {
        *this->layers[0] = this->X->row(i);
        this->forwardProp();
        this->readLayer(0);
        this->readLayer(this->L);
        std::cout << "\n";
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
        *this->layers[0] = this->X->row(i);
        forwardProp();
        out = this->layers[this->L-1]->array();
        J += ((this->Y->row(i).array() * out.log()) + (1 - this->Y->row(i).array())*(1 - out).log()).sum();
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
    for(int l = 0 ; l < L-1 ; l++) {
        *this->weights[l] -= alpha * *gradients[l];
    }
    
    return 0;
}

int NeuralNetwork::backProp() {
    for(int i = 1 ; i < this->L-1 ; i++) {
        this->delta[i]->setZero();
    }

    for(int i = 0 ; i < this->m ; i++){
        *this->layers[0] = this->X->row(i);
        this->forwardProp();
        ping();
        *this->costs[L-1] = (*this->layers[L-1] - this->Y->row(i)).matrix();
        ping("Ap");
        for(int k = L-2 ; k > 1 ; k--) {
            *this->costs[k] = ((*this->costs[k+1] * *this->weights[k]).array() * this->midBLayers[k]->unaryExpr(&NNM::sigmoidp).array()).matrix();
        }
        ping("CCOCKOCKCO");
        for(int k = 0 ; k < L-1 ; k++) {
            std::cout << "k: " << k << std::endl;
            NNM::size(*this->delta[k]);
            NNM::size(this->bLayers[k]->transpose());
            NNM::size(this->layers[k]->transpose());
            NNM::size(*this->costs[k+1]);
            *this->delta[k] +=  this->bLayers[k]->transpose() * *this->costs[k+1];
            *gradients[k] = *delta[k]*this->mi;
        }
        ping();
        for(int k = 0 ; k < this->L-2 ; k++) {
                        
        }
    }
    return 0;
}

int NeuralNetwork::forwardProp() { // Performs forward propagation on the network, updating
    for(int currentLayer = 0 ; currentLayer < L-1 ; currentLayer++) {
        this->bLayers[currentLayer]->block(0,1,1,this->layerSizes[currentLayer]) = *this->layers[currentLayer];
        this->midBLayers[currentLayer+1]->block(0,1,1,this->layerSizes[currentLayer+1]) = (*this->bLayers[currentLayer] * this->weights[currentLayer]->transpose());
        *this->layers[currentLayer+1] = this->midBLayers[currentLayer+1]->block(0,1,1,this->layerSizes[currentLayer+1]).unaryExpr(&NNM::sigmoid);
        
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

    this->m = this->X->rows();
    this->n = this->X->cols();
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
    this->delta.resize(L);

    this->layers.resize(L);
    this->bLayers.resize(L);
    this->midLayers.resize(L);
    this->midBLayers.resize(L);

    for(int l = 0 ; l < this->L-1; l++) {
        this->weights[l] = new Eigen::MatrixXf(this->layerSizes[l+1], this->layerSizes[l]+1);
        this->gradients[l] = new Eigen::MatrixXf(this->layerSizes[l+1], this->layerSizes[l]+1);

        this->delta[l] = new Eigen::MatrixXf(this->layerSizes[l+1], this->layerSizes[l]+1);
        this->costs[l+1] = new Eigen::MatrixXf(1, this->layerSizes[l]+1);

        // Randomise weights to break symmetry
        this->weights[l]->setRandom();
    }
    this->costs[this->L-1]->resize(1,this->layerSizes[this->L-1]);

    for(int l = 0 ; l < this->L; l++) {
        this->layers[l] = new Eigen::MatrixXf(1,layerSizes[l]);
        this->bLayers[l] = new Eigen::MatrixXf(1,layerSizes[l]+1);
        (*this->bLayers[l])(0) = 1;
        this->midLayers[l] = new Eigen::MatrixXf(1,layerSizes[l]);
        this->midBLayers[l] = new Eigen::MatrixXf(1,layerSizes[l]+1);
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