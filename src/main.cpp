#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <functional>
#include <unistd.h>

#include "eigen3/Eigen/Core"

#include "nmath.h"
#include "network.h"
#include "graphics.h"
#include "utils.h"

int main()
{
    bool exit = false;

    srand(time(NULL));

    int iterations = 10000;
    std::vector<float> costs = {1};

	std::mutex bufferLock; // Used to restrict the access to std::vector<float> Costs by each thread
	std::mutex exitLock; // For bool exit
	std::vector<float> Costs; // Tracks the cost associated with each pass	

    Eigen::MatrixXf X, Y;

    X.resize(4,2);
    Y.resize(4,1);

    X.row(0) << 0., 0.;
    X.row(1) << 0., 1.;
    X.row(2) << 1., 0.;
    X.row(3) << 1., 1.;

    Y.row(0) << 1.;
    Y.row(1) << 0.;
    Y.row(2) << 0.;
    Y.row(3) << 0.;

    std::vector<int> sl = {2, 3, 3, 1};
    NeuralNetwork catDogClassifier(sl);
    
    catDogClassifier.setTraining(&X, &Y);   


    *catDogClassifier.layers[0] = X.row(2);
    catDogClassifier.forwardProp();
    catDogClassifier.readLayer(0);
    catDogClassifier.readLayer(catDogClassifier.L-1);


    std::cout << "Cost Before: " << catDogClassifier.cost() << "\n";

    
    std::thread graphing_thread( // Graphs data passed to it by the training thread via the Costs vector
		&grapher,
		std::ref(exit),
		std::ref(costs),
		std::ref(bufferLock), // Mutex to modify Costs
		std::ref(exitLock) // Mutex to modify exit
	);

    bool exitLoop = false;

    for(int i = 0 ; i < iterations && !exitLoop ; i++ ) {
        ping();
        catDogClassifier.gradientDescent();
        ping();

        static float cc;
        cc = catDogClassifier.cost();

        bufferLock.lock();
        costs.push_back(cc);
        bufferLock.unlock();

        if(std::isnan(cc)) {
            std::cout << "NaN detected. Aborting" << std::endl; 
            exitLoop = true;
        }
    }

    exitLock.lock();
    exit = true;
    exitLock.unlock();
    
    std::cout << "Cost After: " << catDogClassifier.cost() << "\n";

	std::cout << "Joining graphing_thread" << std::endl;
	graphing_thread.join();

    *catDogClassifier.layers[0] = X.row(0);
    catDogClassifier.forwardProp();
    catDogClassifier.readLayer(0);
    catDogClassifier.readLayer(catDogClassifier.L-1);
    
    return 0;
}