#ifndef MODEL_FUNCTIONS_H
#define MODEL_FUNCTIONS_H

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <tuple>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>


Eigen::MatrixXd create_lin_mesh(double lSide, int nSide, Eigen::Vector3d cpt, double angle);

std::tuple <Eigen::VectorXd, Eigen::VectorXd> take_reduced_mesh(Eigen::VectorXd phi_big, Eigen::VectorXd dphi_big, int N, int n);
                                  
Eigen::MatrixXd lift_z(Eigen::MatrixXd pos_in, int nx, int ny, double z_sum);

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> compute_l0_linear(Eigen::MatrixXd nodeInitial, int nx, int ny);

typedef struct linmdl LinMdl;
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd> create_model_linear_matrices(LinMdl COM);

std::tuple<LinMdl, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd> init_linear_model(LinMdl MDL, Eigen::MatrixXd posini_XZ);

#endif
