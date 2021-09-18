/*
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
*/

#include "model_functions.h"

using namespace std;

typedef struct linmdl {
  int row;
  int col;
  double mass;
  double grav;
  double dt;
  Eigen::Vector3d stiffness;
  Eigen::Vector3d damping;
  double z_sum;
  Eigen::MatrixXd nodeInitial;
  Eigen::MatrixXd mat_x;
  Eigen::MatrixXd mat_y;
  Eigen::MatrixXd mat_z;
  Eigen::VectorXd coord_ctrl;
  Eigen::VectorXd coord_lc;
} LinMdl;


// ---------------------------------
// -------- MODEL FUNCTIONS --------
// ---------------------------------


// Create a mesh of points in space
Eigen::MatrixXd create_lin_mesh(double lSide, int nSide, Eigen::Vector3d cpt, double angle) {
	
	int MeshLength = nSide*nSide;
	Eigen::MatrixXd MeshPosXZ(MeshLength,3);
	Eigen::MatrixXd MeshPos(MeshLength,3);
	
	for (int i=0; i<MeshLength; i++) {
    MeshPosXZ(i,0) = -lSide/2 + (i%nSide)*lSide/(nSide-1);
    MeshPosXZ(i,1) = 0;
    MeshPosXZ(i,2) = -lSide/2 + (i/nSide)*lSide/(nSide-1);
  }
  
  Eigen::Matrix3d RotM;
  RotM << cos(angle), -sin(angle), 0,
  				sin(angle),  cos(angle), 0,
  				         0,           0, 1;
	
	MeshPos = (RotM * MeshPosXZ.transpose()).transpose();
	for (int i=0; i<MeshLength; i++) {
    MeshPos.row(i) += cpt.transpose();
  }
	
	return MeshPos;
}


// Get coordinates of reduced model from original
tuple <Eigen::VectorXd, Eigen::VectorXd> take_reduced_mesh(Eigen::VectorXd phi_big,
																													 Eigen::VectorXd dphi_big,
                                  												 int N, int n) {

  Eigen::VectorXd redc(n*n);
  for (int i=0; i<n*n; i++) {
    redc(i) = (i%n)*(N-1)/(n-1) + (i/n)*(N-1)/(n-1)*N;
  }
  
  Eigen::VectorXd phi1(n*n*3);
  Eigen::VectorXd dphi1(n*n*3);
  int j=0;
  for (int coord=0; coord<3; coord++) {
  	for (int i=0; i<(n*n); i++) {
  		int redc_i = redc(i);
  		phi1(j) = phi_big(N*N*coord+redc_i);
  		dphi1(j) = dphi_big(N*N*coord+redc_i);
  		j++;
  	}
  }
  
  return make_tuple(phi1, dphi1);
}


// Displace the nodes a given distance in the vertical direction (z)
Eigen::MatrixXd lift_z(Eigen::MatrixXd pos_in, int nx, int ny, double z_sum) {
	Eigen::MatrixXd pos_out(nx*ny,3);
	pos_out.setZero(nx*ny,3);
	
	int i=0;
	
	/*
	  Matlab code iterates the other way (i=1~4 is last row)
	  and creates a 4x4 cell of 1x3 vectors.
	  In any case, the nodes on the last 4 rows of pos_in must not move
	*/
	for (int row=0; row<nx; row++) {
		for (int col=0; col<ny; col++) {
			double d_up = 0;
			if (row!=nx-1) d_up = z_sum;
			
			pos_out.row(i) << pos_in(i,0), pos_in(i,1), pos_in(i,2)+d_up;
			
			i++; 
		}
	}
	
	return pos_out;
	
}


// Compute initial spring length
tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> compute_l0_linear(
                                Eigen::MatrixXd nodeInitial, int nx, int ny) {
                        
  Eigen::MatrixXd mat_x(nx*ny,6);
  Eigen::MatrixXd mat_y(nx*ny,6);
  Eigen::MatrixXd mat_z(nx*ny,6);
  mat_x.setZero(nx*ny,6);
  mat_y.setZero(nx*ny,6);
  mat_z.setZero(nx*ny,6);
  
	int i=0;
	
	for (int row=0; row<nx; row++) {
		int nextRow = row+1;
		int prevRow = row-1;
		for (int col=0; col<ny; col++) {
			int nextCol = col+1;
			int prevCol = col-1;
			
			// Link 3 (Rows are changed)
			if (row < nx-1) {
				Eigen::VectorXd l0 = nodeInitial.row(row*nx+col) - nodeInitial.row(nextRow*nx+col);
				mat_x(i,4) = l0(0);
				mat_y(i,4) = l0(1);
				mat_z(i,4) = l0(2);
			}
			
			// Link 2
			if (col < ny-1) {
				Eigen::VectorXd l0 = nodeInitial.row(row*nx+col) - nodeInitial.row(row*nx+nextCol);
				mat_x(i,2) = l0(0);
				mat_y(i,2) = l0(1);
				mat_z(i,2) = l0(2);
			}
			
			// Link 1 (Rows are changed)
			if (row > 0) {
				Eigen::VectorXd l0 = nodeInitial.row(row*nx+col) - nodeInitial.row(prevRow*nx+col);
				mat_x(i,1) = l0(0);
				mat_y(i,1) = l0(1);
				mat_z(i,1) = l0(2);
			}
			
			// Link 4
			if (col > 0) {
				Eigen::VectorXd l0 = nodeInitial.row(row*nx+col) - nodeInitial.row(row*nx+prevCol);
				mat_x(i,5) = l0(0);
				mat_y(i,5) = l0(1);
				mat_z(i,5) = l0(2);
			}
			
			i++;
		}
	}
	
  return make_tuple(mat_x, mat_y, mat_z);
}


// Creates the matrices to express the system as a linear SS
tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd>
                  create_model_linear_matrices(LinMdl COM) {
                  
	Eigen::Vector3d k  = COM.stiffness;
	Eigen::Vector3d b  = COM.damping;
	double ts = COM.dt;
	double m  = COM.mass;
	double g  = COM.grav;
	int    nx = COM.row;
	int    ny = COM.col;
	int blksz = 3*nx*ny;
	Eigen::VectorXd coord_ctrl = COM.coord_ctrl;
	
	// Define outputs
 	Eigen::MatrixXd A(2*blksz,2*blksz);
 	Eigen::MatrixXd B(2*blksz,2*3);
 	Eigen::VectorXd ext_force(2*blksz);
 	A.setZero(2*blksz,2*blksz);
 	B.setZero(2*blksz,2*3);
 	ext_force.setZero(2*blksz);
	
	// Connectivity matrix (nodes numbered from bottom to top and left to right)
	// Modified from Matlab: not hardcoded
	Eigen::MatrixXd conn(nx*ny,nx*ny);
	for (int i=0; i<nx*ny; i++) {
	  if(i%ny != 0) conn(i, i-1) = -1;
	  if(i%ny != ny-1) conn(i, i+1) = -1;
	  if(i/ny != 0) conn(i, i-ny) = -1;
	  if(i/ny != nx-1) conn(i, i+ny) = -1;
	}
	// Remove connections on controlled nodes (assumes 2 first coord_ctrl)
  conn.row(coord_ctrl(0)).setZero();
  conn.row(coord_ctrl(1)).setZero();

	// Diagonal as -sum of connections
	Eigen::MatrixXd diago(nx*ny,nx*ny);
	diago = -1*conn.rowwise().sum().asDiagonal();

	// "Unitary force"
	Eigen::MatrixXd F(nx*ny,nx*ny);
	F = diago+conn;

	// Spring forces in each direction (with its stiffness)
	Eigen::MatrixXd Fp(blksz,blksz);
	Fp.setZero(blksz,blksz);

	Fp.block(0,0,nx*ny,nx*ny) = k(0)*F;
	Fp.block(nx*ny,nx*ny,nx*ny,nx*ny) = k(1)*F;
	Fp.block(2*nx*ny,2*nx*ny,nx*ny,nx*ny) = k(2)*F;

	// Damper forces in each direction (with its damping)
	Eigen::MatrixXd Fv(blksz,blksz);
	Fv.setZero(blksz,blksz);

	Fv.block(0,0,nx*ny,nx*ny) = b(0)*F;
	Fv.block(nx*ny,nx*ny,nx*ny,nx*ny) = b(1)*F;
	Fv.block(2*nx*ny,2*nx*ny,nx*ny,nx*ny) = b(2)*F;

	/*
	 System equations:
		x(k+1) = x(k) + v(k)*dt
		v(k+1) = v(k) + a(k)*dt
		
	 Where: a(k) = (1/m) * F_total = (1/m) * (K*x(k) + b*v(k))
	 Thus:
		[pos_next] =         I*[pos] +        dt*I*[vel]
		[vel_next] = (dt/m)*Fp*[pos] + (I+dt/m*Fv)*[vel]
  */

	A.block(0,0,blksz,blksz) = Eigen::MatrixXd::Identity(blksz,blksz);	
	A.block(0,blksz,blksz,blksz) = ts * Eigen::MatrixXd::Identity(blksz,blksz);
	A.block(blksz,0,blksz,blksz) = (ts/m)*Fp;
  A.block(blksz,blksz,blksz,blksz) = (ts/m)*Fv + Eigen::MatrixXd::Identity(blksz,blksz);
	
	// Initial length matrices
	Eigen::VectorXd mx(nx*ny);
	Eigen::VectorXd my(nx*ny);
	Eigen::VectorXd mz(nx*ny);
	mx.setZero(nx*ny);
	my.setZero(nx*ny);
	mz.setZero(nx*ny);
	
	mx = COM.mat_x.rowwise().sum();
	my = COM.mat_y.rowwise().sum();
	mz = COM.mat_z.rowwise().sum();
	
	Eigen::VectorXd grav(2*blksz);
	grav.setZero(2*blksz);
	grav.segment(2*blksz-nx*ny, nx*ny) = -g*Eigen::VectorXd::Ones(nx*ny);
	
	Eigen::VectorXd l0_springs(2*blksz);
	l0_springs.setZero(2*blksz);
	l0_springs.segment(blksz, nx*ny) = -k(0)/m * mx;
	l0_springs.segment(blksz+nx*ny, nx*ny) = -k(1)/m * my;
	l0_springs.segment(blksz+2*nx*ny, nx*ny) = -k(2)/m * mz;
	
	ext_force = grav + l0_springs;
	
	// A: The upper corners are set to 0 because they are fixed
	// B: Control matrix has only 1 on the upper corners
	// f: Also 0 on controlled coordinates
	for (unsigned int i=0; i<coord_ctrl.size(); i++) {
	  A.block(coord_ctrl(i),blksz, 1,blksz) = Eigen::MatrixXd::Zero(1,blksz);
	  A.block(coord_ctrl(i)+blksz,0, 1,2*blksz) = Eigen::MatrixXd::Zero(1,2*blksz);
	  
	  B(coord_ctrl(i), i) = 1;
	  
	  ext_force(coord_ctrl(i)+blksz) = 0;
	}
	
	return make_tuple(A, B, ext_force);
	
}


// Initialize linear model (sequence of operations with previous functions)
tuple<LinMdl, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd> 
										init_linear_model(LinMdl MDL, Eigen::MatrixXd posini_XZ) {

	int nx = MDL.row;
	int ny = MDL.col;	
	int MdlLength = nx*ny;
  
  // Initial position of the Model nodes
  Eigen::MatrixXd nodeInitial(MdlLength,3);
  nodeInitial = lift_z(posini_XZ, nx, ny, MDL.z_sum);
  MDL.nodeInitial = nodeInitial;
  
  // Find initial spring length in each direction (x,y,z)
  Eigen::MatrixXd mat_x(MdlLength,6);
  Eigen::MatrixXd mat_y(MdlLength,6);
  Eigen::MatrixXd mat_z(MdlLength,6);
  mat_x.setZero(MdlLength,6);
  mat_y.setZero(MdlLength,6);
  mat_z.setZero(MdlLength,6);
  
  tie(mat_x, mat_y, mat_z) = compute_l0_linear(nodeInitial, nx, ny);
  MDL.mat_x = mat_x;
 	MDL.mat_y = mat_y;
 	MDL.mat_z = mat_z;
 	
 	// Find linear matrices
 	Eigen::MatrixXd A(6*MdlLength,6*MdlLength);
 	Eigen::MatrixXd B(6*MdlLength,6);
 	Eigen::VectorXd ext_force(6*MdlLength);
 	A.setZero(6*MdlLength,6*MdlLength);
 	B.setZero(6*MdlLength,6);
 	ext_force.setZero(6*MdlLength);
 	
 	tie(A, B, ext_force) = create_model_linear_matrices(MDL);
 	
 	return make_tuple(MDL, A, B, ext_force);
}





