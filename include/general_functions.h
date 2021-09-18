#ifndef GENERAL_FUNCTIONS_H
#define GENERAL_FUNCTIONS_H

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>


std::vector<std::string> split(std::string s, char c);

void saveAsCSV(std::string name, Eigen::MatrixXd matrix);

Eigen::MatrixXd getCSVcontent(std::string filename);

#endif
