#ifndef _ROOT_SRC_NMATH_H
#define _ROOT_SRC_NMATH_H

#include <cmath>

#include "eigen3/Eigen/Core"

namespace NNM {
    float sigmoid(float z);
    float sigmoidp(float z);
    void size(Eigen::MatrixXf mat);
}

#endif