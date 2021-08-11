#ifndef _ROOT_SRC_NETWORK_H
#define _ROOT_SRC_NETWORK_H

#include <vector>
#include <list>

#include "eigen3/Eigen/Core"

#include "nmath.h"

#define WEIGHTS 0
#define LAYERSIZES 1
#define LAYERS 2

class NeuralNetwork{
    private:
        /*
            Calculation variables
        */
       float mi = 0; // holds 1/m


        // An STL vector which contains the value of each neuron of each layer before it is passed through the activation function
        std::vector<Eigen::MatrixXf*> midLayers;
        // Same as midLayers, but also has a bias unit
        std::vector<Eigen::MatrixXf*> midBLayers;

    public:
        /*
            Variables
        */
        
        // Training Variables
        Eigen::MatrixXf* X;
        Eigen::MatrixXf* Y;

        int m = 0; // holds the amount of training examples
        int n = 0; // holds the amount of features

        // Hyper-Parameteres
        float alpha = 1; // Learning rate
        float lambda = 0; // Regularisation

        //Layered data
        // Holds pointers to the matrix of weights for each layer
        std::vector<Eigen::MatrixXf*> weights;
        // Holds the cost of each connection
        std::vector<Eigen::MatrixXf*> delta;
        // Holds pointers to the matrix of unaveraged and unregularised gradients for each layer
        std::vector<Eigen::MatrixXf*> costs;
        // Holds pointers to the matrix of gradients for each layer
        std::vector<Eigen::MatrixXf*> gradients;
        // Holds pointers to the matrix of neurons for each layer.
        std::vector<Eigen::MatrixXf*> layers;
        // Same as layers, but with an added bias unit. Used as a base for calculating the next layer
        std::vector<Eigen::MatrixXf*> bLayers;

        

        // An STL vector which contains the unit count for each layer
        std::vector<int> layerSizes; 

        //      Layer information

        int L; // Layer count
        int K; // Output unit count

        /*
            Methods
        */

        void gradientChecking();
        void readLayer(int l);
        void readOutputs();
        void checkAll();
        float cost();
        float cost(std::vector<Eigen::MatrixXf>& theta);
        int gradientDescent();
        int forwardProp();
        int forwardProp(std::vector<Eigen::MatrixXf>& theta);        
        int backProp();
        int setTraining(Eigen::MatrixXf* inputs, Eigen::MatrixXf* outputs); 

        /*  Constructor
            - sl is a vector of integers, where sl[l] = the amount of neurons in layer l
            - new keyword is used when initialising this->weights and this->layers, and must be deallocated in the destructor
        */
        NeuralNetwork(std::vector<int>& sl);

        /*  Destructor
            - Explicitly deallocates memory for weights and layers
        */
        ~NeuralNetwork();
};

#endif