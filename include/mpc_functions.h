#ifndef MPC_FUNCTIONS_H
#define MPC_FUNCTIONS_H

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <tuple>
#include <casadi/casadi.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

casadi::SX normalize3(casadi::SX in);

casadi::SX get_adaptive_Q(casadi::SX Rp, casadi::SX x0, Eigen::VectorXd coord_l);

#endif
