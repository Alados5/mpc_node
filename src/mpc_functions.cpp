/*
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
*/

#include "mpc_functions.h"

using namespace std;
using namespace casadi;

// Normalize a vector of 3 components (abs/norm)
SX normalize3(SX in) {

	return abs(in)/(sqrt(in(0)*in(0)+in(1)*in(1)+in(2)*in(2)) + 1e-6);
	
}


// Weights (adaptive) calculation: direction from actual to desired position
SX get_adaptive_Q(SX Rp, SX x0, Eigen::VectorXd coord_l) {

	// Left-Right nodes
	SX lc_dist = Rp(Slice(),-1);
	SX ln = SX::vertcat({lc_dist(0), lc_dist(2), lc_dist(4)});
	SX rn = SX::vertcat({lc_dist(1), lc_dist(3), lc_dist(5)});
	for (int i=0; i<3; i++) {
		ln(i) -= x0(coord_l(2*i));
		rn(i) -= x0(coord_l(2*i+1));
	}
	ln = normalize3(ln);
	rn = normalize3(rn);
	
	// Final matrix
	SX Qv = SX::vertcat({ln(0), rn(0), ln(1), rn(1), ln(2), rn(2)});
	SX Q = SX::diag(Qv);
	
	return Q;
	
}




