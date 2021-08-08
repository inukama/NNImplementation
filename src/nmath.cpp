#include <cmath>
#include <iostream>

#include "eigen3/Eigen/Core"

#include "nmath.h"

namespace NNM {
    float sigmoid(float z) {
        return std::tanh(z*0.5)*0.5 + 0.5;
    }
    float sigmoidp(float z) {
        return 1/(2*std::cosh(z)+2);
    }
    void size(Eigen::MatrixXf mat) {
        std::cout << mat.rows() << " " << mat.cols() << "\n";
    }
}