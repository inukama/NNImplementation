#ifndef _ROOT_SRC_READER_H
#define _ROOT_SRC_READER_H

#include <iostream>
#include <fstream>
#include <istream>

#include "eigen3/Eigen/Core"

void mnistToMatrix(Eigen::MatrixXf& X, Eigen::MatrixXf& Y, const char* imageSetLocation, const char* labelLocation);

#endif