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

    int iterations = 100000;
    std::vector<float> costs = {1};

	std::mutex bufferLock; // Used to restrict the access to std::vector<float> Costs by each thread
	std::mutex exitLock; // For bool exit
	std::vector<float> Costs; // Tracks the cost associated with each pass	

    Eigen::MatrixXf X, Y;

    X.resize(2,4);
    Y.resize(1,4);

    X.col(0) << 0., 0.;
    X.col(1) << 0., 1.;
    X.col(2) << 1., 0.;
    X.col(3) << 1., 1.;

    Y.col(0) << 1.;
    Y.col(1) << 1.;
    Y.col(2) << 0.;
    Y.col(3) << 0.;

    std::vector<int> sl = {2, 20,76, 1};
    NeuralNetwork catDogClassifier(sl);

    catDogClassifier.setTraining(&X, &Y);   

    std::cout << "Cost Before: " << catDogClassifier.cost() << "\n";
    std::cout << "Beginning Training..." << std::endl << std::endl;

    
    std::thread graphing_thread( // Graphs data passed to it by the training thread via the Costs vector
		&grapher,
		std::ref(exit),
		std::ref(costs),
		std::ref(bufferLock), // Mutex to modify Costs
		std::ref(exitLock) // Mutex to modify exit
	);

    bool exitLoop = false;

    Eigen::MatrixXf testmat = *catDogClassifier.weights[0];

    for(int i = 0 ; i < iterations && !exitLoop ; i++ ) {
        catDogClassifier.gradientDescent();

        catDogClassifier.alpha *= 1.0000;

        static float cc;
        cc = catDogClassifier.cost();

        bufferLock.lock();
        costs.push_back(cc);
        bufferLock.unlock();

        if(std::isnan(cc)) {
            std::cout << "NaN detected. Aborting at alpha = " << catDogClassifier.alpha << std::endl;
            exitLoop = true;
        }
        if(cc <= 0.000001) {
            std::cout << "Gradient descent has converged. Exiting." << std::endl;
            exitLoop = true;
        }
    }

    exitLock.lock();
    exit = true;
    exitLock.unlock();
    
    std::cout << "Training ended" << std::endl;
    std::cout << "Cost After: " << catDogClassifier.cost() << "\n";

	std::cout << "Joining graphing_thread" << std::endl;
	graphing_thread.join();

    catDogClassifier.checkAll();

    std::cout << "Ending program" << std::endl;
    
    return 0;
}