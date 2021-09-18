#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <casadi/casadi.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include "general_functions.h"
#include "model_functions.h"
#include "mpc_functions.h"
#include <ros/ros.h>
#include "cartesian_msgs/CartesianCommand.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/Quaternion.h"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <mpc_pkg/TwoPoses.h>
#include <mpc_pkg/TwoPoints.h>
#include <mpc_vision/SOMstate.h>
//#include <std_srvs/Empty.h>


using namespace std;
using namespace casadi;


// GLOBAL VARIABLES
string datapath = "/home/robot/Desktop/ALuque/mpc_ros/src/mpc_node/data/";
int NTraj = 0;
int nCOM = 0;
int nSOM = 0;
int Hp = 0;
double Ts = 0.00;
double W_Q = 1.0;
double W_R = 0.2;
double ubound = 0.010;
double gbound = 0.0;
bool opt_du = true;
bool opt_Qa = false;
Eigen::Vector3d tcp_offset_local(0.0, 0.0, 0.09);

// Needed to update Vision SOM
int MaxVSteps = 10;
double MaxVDiff = 0.03;
double Wv = 0.0;
Eigen::MatrixXd A_SOM;
Eigen::MatrixXd B_SOM;
Eigen::VectorXd f_SOM;
Eigen::VectorXd state_SOM;
Eigen::VectorXd SOM_nodes_ctrl(2);
Eigen::VectorXd SOM_coord_ctrl(6);
Eigen::MatrixXd uSOMhist(6,MaxVSteps);
int uhistID = 0;
// ----------------


// Linear model structure
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



// ROS SUBSCRIBERS - CALLBACK FUNCTIONS
void somstateReceived(const mpc_vision::SOMstate& msg) {
	
	double Vision_t = msg.header.stamp.toSec();
	int nMesh = msg.size;
	int Meshlength = nMesh*nMesh;
	
	Eigen::VectorXf Vision_SOMstatef = Eigen::VectorXf::Map(msg.states.data(), msg.states.size());
  Eigen::VectorXd Vision_SOMstate = Vision_SOMstatef.cast<double>();

	Eigen::VectorXd VSOMpos(3*nSOM*nSOM);
  Eigen::VectorXd VSOMvel(3*nSOM*nSOM);
  Eigen::VectorXd VSOMstate(6*nSOM*nSOM);
  tie(VSOMpos, VSOMvel) = take_reduced_mesh(Vision_SOMstate.segment(0,3*Meshlength),
  																					Vision_SOMstate.segment(3*Meshlength,3*Meshlength),
  																					nMesh, nSOM);
  VSOMstate << VSOMpos, VSOMvel;
  																							
  //double Vision_dt = ros::Time::now().toSec() - Vision_t;
  ros::Duration Vision_dt = ros::Time::now() - msg.header.stamp;
  
  // Update the captured SOMstate to current time (dt/Ts steps)
  int update_steps = Vision_dt.toSec()/Ts;
  update_steps = max(update_steps, 0);
  if (update_steps > MaxVSteps) {
  	update_steps = MaxVSteps;
  	ROS_WARN_STREAM("Delay for captured Mesh too high! Ignoring data.");
  	return;
  }
  
  // Needed variables
  Eigen::VectorXd uSOM_Vti(6);
  Eigen::VectorXd ulin_Vti(6);
  Eigen::VectorXd urot_Vti(6);
  Eigen::MatrixXd ulin2_Vti(3,2);
  Eigen::MatrixXd urot2_Vti(3,2);
  Eigen::Vector3d Vcloth_x;
  Eigen::Vector3d Vcloth_y;
  Eigen::Vector3d Vcloth_z;
  Eigen::Matrix3d VRcloth;
  Eigen::MatrixXd pos_ini_VSOM(nSOM*nSOM,3);
  Eigen::MatrixXd vel_ini_VSOM(nSOM*nSOM,3);
  Eigen::MatrixXd pos_ini_VSOM_rot(nSOM*nSOM,3);
  Eigen::MatrixXd vel_ini_VSOM_rot(nSOM*nSOM,3);
  Eigen::MatrixXd pos_nxt_VSOM_rot(nSOM*nSOM,3);
  Eigen::MatrixXd vel_nxt_VSOM_rot(nSOM*nSOM,3);
  Eigen::MatrixXd pos_nxt_VSOM(nSOM*nSOM,3);
  Eigen::MatrixXd vel_nxt_VSOM(nSOM*nSOM,3);
  Eigen::VectorXd VSOMstate_rot(6*nSOM*nSOM);
	
	// Iterate until current time
	for (int Vti=-update_steps; Vti<0; Vti++) {
		
		// Real history ID
		int uhistIDi = uhistID + Vti;
		if (uhistIDi < 0) uhistIDi+=MaxVSteps;
		
		// Get uSOM from that instant, skip 0s
		uSOM_Vti = uSOMhist.col(uhistIDi);
		if (uSOM_Vti == Eigen::VectorXd::Zero(6)) continue;
		
		// Get linear control actions (displacements)
		for (int ucoordi=0; ucoordi<6; ucoordi++) {
			ulin_Vti(ucoordi) = uSOM_Vti(ucoordi) - VSOMstate(SOM_coord_ctrl(ucoordi));
		}
		ulin2_Vti.col(0) << ulin_Vti(0), ulin_Vti(2), ulin_Vti(4);
		ulin2_Vti.col(1) << ulin_Vti(1), ulin_Vti(3), ulin_Vti(5);
		
		// Obtain Vcloth base
		Vcloth_x << VSOMstate(SOM_coord_ctrl(1))-VSOMstate(SOM_coord_ctrl(0)),
			          VSOMstate(SOM_coord_ctrl(3))-VSOMstate(SOM_coord_ctrl(2)),
			          VSOMstate(SOM_coord_ctrl(5))-VSOMstate(SOM_coord_ctrl(4));
		Vcloth_y << -Vcloth_x(1), Vcloth_x(0), 0;                 
		Vcloth_z = Vcloth_x.cross(Vcloth_y);
		
		Vcloth_x = Vcloth_x/Vcloth_x.norm();
		Vcloth_y = Vcloth_y/Vcloth_y.norm();
		Vcloth_z = Vcloth_z/Vcloth_z.norm();
		VRcloth << Vcloth_x, Vcloth_y, Vcloth_z;
		
		// Linear SOM uses local base variables (rot)
		pos_ini_VSOM.col(0) = VSOMstate.segment(0, nSOM*nSOM);
		pos_ini_VSOM.col(1) = VSOMstate.segment(1*nSOM*nSOM, nSOM*nSOM);
		pos_ini_VSOM.col(2) = VSOMstate.segment(2*nSOM*nSOM, nSOM*nSOM);
		vel_ini_VSOM.col(0) = VSOMstate.segment(3*nSOM*nSOM, nSOM*nSOM);
		vel_ini_VSOM.col(1) = VSOMstate.segment(4*nSOM*nSOM, nSOM*nSOM);
		vel_ini_VSOM.col(2) = VSOMstate.segment(5*nSOM*nSOM, nSOM*nSOM);
	
		pos_ini_VSOM_rot = (VRcloth.inverse() * pos_ini_VSOM.transpose()).transpose();
		vel_ini_VSOM_rot = (VRcloth.inverse() * vel_ini_VSOM.transpose()).transpose();
		
		VSOMstate_rot << pos_ini_VSOM_rot.col(0),
				             pos_ini_VSOM_rot.col(1),
				             pos_ini_VSOM_rot.col(2),
				             vel_ini_VSOM_rot.col(0),
				             vel_ini_VSOM_rot.col(1),
				             vel_ini_VSOM_rot.col(2);
				             
		// Rotate control actions from history
		urot2_Vti = VRcloth.inverse() * ulin2_Vti;
		urot_Vti << urot2_Vti.row(0).transpose(),
								urot2_Vti.row(1).transpose(),
								urot2_Vti.row(2).transpose();
				             
		// Simulate a step
		VSOMstate_rot = A_SOM*VSOMstate_rot + B_SOM*urot_Vti + Ts*f_SOM; 
	
		// Convert back to global axes
		pos_nxt_VSOM_rot.col(0) = VSOMstate_rot.segment(0, nSOM*nSOM);
		pos_nxt_VSOM_rot.col(1) = VSOMstate_rot.segment(1*nSOM*nSOM, nSOM*nSOM);
		pos_nxt_VSOM_rot.col(2) = VSOMstate_rot.segment(2*nSOM*nSOM, nSOM*nSOM);
		vel_nxt_VSOM_rot.col(0) = VSOMstate_rot.segment(3*nSOM*nSOM, nSOM*nSOM);
		vel_nxt_VSOM_rot.col(1) = VSOMstate_rot.segment(4*nSOM*nSOM, nSOM*nSOM);
		vel_nxt_VSOM_rot.col(2) = VSOMstate_rot.segment(5*nSOM*nSOM, nSOM*nSOM);
	
		pos_nxt_VSOM = (VRcloth * pos_nxt_VSOM_rot.transpose()).transpose();
		vel_nxt_VSOM = (VRcloth * vel_nxt_VSOM_rot.transpose()).transpose();
		VSOMstate << pos_nxt_VSOM.col(0),
			           pos_nxt_VSOM.col(1),
			           pos_nxt_VSOM.col(2),
			           vel_nxt_VSOM.col(0),
			           vel_nxt_VSOM.col(1),
			           vel_nxt_VSOM.col(2);
  }
  
  // Filter outlying and incorrect mesh data
  Eigen::VectorXd dSOMstate = (VSOMstate - state_SOM).cwiseAbs();
	double mean_dpos = dSOMstate.segment(0,3*nSOM*nSOM).sum()/(3*nSOM*nSOM);
	if (mean_dpos > MaxVDiff) {
	  ROS_WARN_STREAM("Vision data is too far away from simulation. Ignoring.");
	  return;
	}

	// DEBUGGING
	ROS_WARN_STREAM("SOM state received!\n"<<
	                "- Update steps: "<<update_steps <<endl<<
	                "- xVSOM(1)i: "<<VSOMpos(1)<<endl<<
	                "- xVSOM(1)f: "<<VSOMstate(1)<<endl<<
	                "- xSOM(1):  "<<state_SOM(1)<<endl<<
	                "- Avg dpos: "<<mean_dpos<<endl);
	                
	// Update SOM states (weighted average)
	ROS_INFO_STREAM("Updated SOM states with Vision data");
	state_SOM = state_SOM*(1-Wv) + VSOMstate*Wv;
	
}




// ---------------------------
// -------- MAIN PROG --------
// ---------------------------

int main(int argc, char **argv) {

	// Initialize the ROS system and become a node.
  ros::init(argc, argv, "mpc_cl_node");
  ros::NodeHandle rosnh;
  
	// Define client objects to all services
  //ros::ServiceClient clt_foo = rosnh.serviceClient<node::Service>("service_name");
  //ros::service::waitForService("service_name");
  
  // Define Publishers
  ros::Publisher pub_usom = rosnh.advertise<mpc_pkg::TwoPoints>
                            ("mpc_controller/u_SOM", 1000);
  ros::Publisher pub_utcp = rosnh.advertise<geometry_msgs::PoseStamped>
                            ("mpc_controller/u_TCP", 1000);
  ros::Publisher pub_uwam = rosnh.advertise<cartesian_msgs::CartesianCommand>
                            ("iri_wam_controller/CartesianControllerNewGoal", 1000);
  
  // Define Subscribers
  ros::Subscriber sub_somstate = rosnh.subscribe("mpc_controller/state_SOM",
                                                 1000, &somstateReceived);
  
  // Get parameters from launch
  if(!rosnh.getParam("/mpc_cl_node/datapath", datapath)) {
  	ROS_ERROR("Need to define the datapath (where reference csv files are) parameter.");
  }
  if(!rosnh.getParam("/mpc_cl_node/NTraj", NTraj)) {
  	ROS_ERROR("Need to define the NTraj (reference trajectory number) parameter.");
  }
  if(!rosnh.getParam("/mpc_cl_node/nSOM", nSOM)) {
  	ROS_ERROR("Need to define the nSOM (SOM mesh side size) parameter.");
  }
  if(!rosnh.getParam("/mpc_cl_node/nCOM", nCOM)) {
  	ROS_ERROR("Need to define the nCOM (COM mesh side size) parameter.");
  }
  if(!rosnh.getParam("/mpc_cl_node/Hp", Hp)) {
  	ROS_ERROR("Need to define the Hp (prediction horizon) parameter.");
  }
  if(!rosnh.getParam("/mpc_cl_node/Ts", Ts)) {
  	ROS_ERROR("Need to define the Ts (sample time) parameter.");
  }
  if(!rosnh.getParam("/mpc_cl_node/Wv", Wv)) {
  	ROS_ERROR("Need to define the Wv (weight of the Vision feedback) parameter.");
  }
  
  
  
 	// --------------------------------
	// 1. INITIAL PARAMETERS DEFINITION
	// --------------------------------
	
	// Load trajectories to follow
  Eigen::MatrixXd phi_l_Traj = getCSVcontent(datapath+"ref_"+to_string(NTraj)+"L.csv");
  Eigen::MatrixXd phi_r_Traj = getCSVcontent(datapath+"ref_"+to_string(NTraj)+"R.csv");
	
	// Get implied cloth size and position
	Eigen::MatrixXd dphi_corners1 = phi_r_Traj.row(0) - phi_l_Traj.row(0);
	double lCloth = dphi_corners1.norm();
  Eigen::Vector3d cCloth;
  cCloth = (phi_r_Traj.row(0) + phi_l_Traj.row(0))/2;
  cCloth(2) += lCloth/2;
  double aCloth = atan2(dphi_corners1(1), dphi_corners1(0));
  
  
  // Initialization of model variables
  LinMdl COM;
  LinMdl SOM;
  
  // Load parameter table and get row(s) for COM/SOM
  // LUT cols: MdlSz, Ts, theta[7]
  Eigen::MatrixXd thetaLUT = getCSVcontent(datapath+"ThetaMdl_LUT.csv");
  bool COMparams = false;
  bool SOMparams = false;
  for (int LUTi=0; LUTi<thetaLUT.rows(); LUTi++) {
		if (thetaLUT(LUTi, 1) != Ts) continue;
		
		Eigen::MatrixXd theta_i = thetaLUT.block(LUTi,2, 1,7);
		
		// Set model parameters
		if (thetaLUT(LUTi, 0) == nSOM) {
			if (SOMparams) ROS_WARN_STREAM("Several rows match the same SOM parameters!");
			
		  SOM.stiffness << theta_i(0), theta_i(1), theta_i(2);
  		SOM.damping   << theta_i(3), theta_i(4), theta_i(5);
  		SOM.z_sum      = theta_i(6);
  		SOMparams = true;
		}
		
		if (thetaLUT(LUTi, 0) == nCOM) {
			if (COMparams) ROS_WARN_STREAM("Several rows match the same COM parameters!");
			
		  COM.stiffness << theta_i(0), theta_i(1), theta_i(2);
  		COM.damping   << theta_i(3), theta_i(4), theta_i(5);
  		COM.z_sum      = theta_i(6);
  		COMparams = true;
		}
		
  }
  // If no row matched the settings
  if (!SOMparams || !COMparams) {
  	ROS_WARN_STREAM("No rows match the parameters! Setting everything to default.");
  	
  	nSOM = 4;
  	nCOM = 4;
  	Ts = 0.01;
  	
		SOM.stiffness << -300.0, -13.5, -225.0;
  	SOM.damping   << -4.0, -2.5, -4.0;
  	SOM.z_sum      =  0.03;

		COM.stiffness << -300.0, -13.5, -225.0;
  	COM.damping   << -4.0, -2.5, -4.0;
  	COM.z_sum      =  0.03;
	}
	
	
  // Rest of COM Definition
  int nxC  = nCOM;
  int nyC  = nCOM;
  COM.row  = nxC;
  COM.col  = nyC;
  COM.mass = 0.1;
  COM.grav = 9.8;
  COM.dt   = Ts;
  int COMlength = nxC*nyC;

  // Important Coordinates (upper/lower corners in x,y,z)
  Eigen::VectorXd COM_nodes_ctrl(2);
	Eigen::VectorXd COM_coord_ctrl(6);
	COM_nodes_ctrl << nyC*(nxC-1), nyC*nxC-1;
	COM_coord_ctrl << COM_nodes_ctrl(0), COM_nodes_ctrl(1),
										COM_nodes_ctrl(0)+nxC*nyC, COM_nodes_ctrl(1)+nxC*nyC,
										COM_nodes_ctrl(0)+2*nxC*nyC, COM_nodes_ctrl(1)+2*nxC*nyC;
	COM.coord_ctrl = COM_coord_ctrl;
	Eigen::VectorXd COM_coord_lc(6);
	COM_coord_lc << 0, nyC-1, nxC*nyC, nxC*nyC+nyC-1, 2*nxC*nyC, 2*nxC*nyC+nyC-1;
	COM.coord_lc = COM_coord_lc;
	
	
	// Rest of SOM Definition. Linear model!
  int nxS  = nSOM;
  int nyS  = nSOM;
  SOM.row  = nxS;
  SOM.col  = nyS;
  SOM.mass = 0.1;
  SOM.grav = 9.8;
  SOM.dt   = Ts;
  int SOMlength = nxS*nyS;

	// Important Coordinates (upper/lower corners in x,y,z)
  SOM_nodes_ctrl << nyS*(nxS-1), nyS*nxS-1;
	SOM_coord_ctrl << SOM_nodes_ctrl(0), SOM_nodes_ctrl(1),
										SOM_nodes_ctrl(0)+nxS*nyS, SOM_nodes_ctrl(1)+nxS*nyS,
										SOM_nodes_ctrl(0)+2*nxS*nyS, SOM_nodes_ctrl(1)+2*nxS*nyS;
  SOM.coord_ctrl = SOM_coord_ctrl;
	Eigen::VectorXd SOM_coord_lc(6);
	SOM_coord_lc << 0, nyS-1, nxS*nyS, nxS*nyS+nyS-1, 2*nxS*nyS, 2*nxS*nyS+nyS-1;
  SOM.coord_lc = SOM_coord_lc;
  
  
  // SOM Initialization: Mesh on XZ plane
  Eigen::MatrixXd pos(SOMlength,3);
  pos = create_lin_mesh(lCloth, nSOM, cCloth, aCloth);

  
  // Define initial position of the nodes (for ext_force)
  // Second half of the vector is velocities (initial = 0)
  Eigen::VectorXd x_ini_SOM(2*3*nxS*nyS);
  x_ini_SOM.setZero(2*3*nxS*nyS);

  x_ini_SOM.segment(0, SOMlength) = pos.col(0);
  x_ini_SOM.segment(SOMlength, SOMlength) = pos.col(1);
  x_ini_SOM.segment(2*SOMlength, SOMlength) = pos.col(2);
  
  // Reduce initial SOM position to COM size if necessary
  Eigen::VectorXd reduced_pos(3*COMlength);
  Eigen::VectorXd reduced_vel(3*COMlength);
  tie(reduced_pos, reduced_vel) = take_reduced_mesh(x_ini_SOM.segment(0,3*SOMlength),
  																									x_ini_SOM.segment(3*SOMlength,3*SOMlength),
  																									nxS, nxC);
  Eigen::VectorXd x_ini_COM(2*3*COMlength);
  x_ini_COM.setZero(2*3*COMlength);
  x_ini_COM.segment(0,3*COMlength) = reduced_pos;
  
  // Rotate initial COM and SOM positions to XZ plane
  Eigen::Matrix3d RCloth_ini;
  RCloth_ini << cos(aCloth), -sin(aCloth), 0,
  				      sin(aCloth),  cos(aCloth), 0,
  				               0,             0, 1;
  				               
  Eigen::MatrixXd posCOM(COMlength,3);
	Eigen::MatrixXd posSOM_XZ(SOMlength,3);
	Eigen::MatrixXd posCOM_XZ(COMlength,3);
	posCOM.col(0) = x_ini_COM.segment(0, COMlength);
	posCOM.col(1) = x_ini_COM.segment(1*COMlength, COMlength);
	posCOM.col(2) = x_ini_COM.segment(2*COMlength, COMlength);	
	
	posSOM_XZ = (RCloth_ini.inverse() * pos.transpose()).transpose();
	posCOM_XZ = (RCloth_ini.inverse() * posCOM.transpose()).transpose();
  

 	// Get the linear model for the COM
 	Eigen::MatrixXd A(6*COMlength,6*COMlength);
 	Eigen::MatrixXd B(6*COMlength,6);
 	Eigen::VectorXd ext_force(6*COMlength);
 	A.setZero(6*COMlength,6*COMlength);
 	B.setZero(6*COMlength,6);
 	ext_force.setZero(6*COMlength);
 	
	tie(COM, A, B, ext_force) = init_linear_model(COM, posCOM_XZ);
	
	
	// Get the linear model for the SOM
 	A_SOM.resize(6*SOMlength, 6*SOMlength);
 	B_SOM.resize(6*SOMlength, 6);
 	f_SOM.resize(6*SOMlength);
 	A_SOM.setZero(6*SOMlength,6*SOMlength);
 	B_SOM.setZero(6*SOMlength,6);
 	f_SOM.setZero(6*SOMlength);
 	
	tie(SOM, A_SOM, B_SOM, f_SOM) = init_linear_model(SOM, posSOM_XZ);
	
  
  // INITIAL INFO
  ROS_INFO_STREAM("\n- Executing Reference Trajectory: "<<NTraj<<
  								"\n- Reference Trajectory has "<<phi_l_Traj.rows() <<" points."<<endl<<
  								"\n- Sample Time (s): \t"<<Ts<<
  								"\n- Prediction Horizon: \t"<<Hp<<
  								"\n- Model sizes: \tnSOM="<<nSOM<<", nCOM="<<nCOM<<endl<<
									"\n- Cloth Side Length (m): \t "<<lCloth<<
                  "\n- Cloth Initial Center: \t"<<cCloth.transpose()<<
                  "\n- Cloth Initial Angle (rad): \t "<<aCloth<<endl);
  
  
	// ----------------------------------
	// 2. OPTIMIZATION PROBLEM DEFINITION
	// ----------------------------------
	
	// Declare model variables
	int n_states = 2*3*COMlength;
	SX xpos = SX::sym("pos", 3*COMlength,Hp+1);
	SX xvel = SX::sym("vel", 3*COMlength,Hp+1);
	SX x = vertcat(xpos, xvel);
	SX u = SX::sym("u", 6,Hp);
	
	// Convert eigen matrices to Casadi matrices
	DM A_DM = DM::zeros(n_states, n_states);
	memcpy(A_DM.ptr(), A.data(), sizeof(double)*n_states*n_states);
	
	DM B_DM = DM::zeros(n_states, 6);
	memcpy(B_DM.ptr(), B.data(), sizeof(double)*n_states*6);
	
	DM f_DM = DM::zeros(n_states, 1);
	memcpy(f_DM.ptr(), ext_force.data(), sizeof(double)*n_states);
	
	// Initial parameters of the optimization problem
	SX P = SX::sym("P", 2+6+3, max(n_states, Hp+1));
	SX x0 = P(0,Slice()).T();
	SX u0 = P(1,Slice(0,6)).T();
	SX Rp = P(Slice(2,8), Slice(0,Hp+1));
	//SX d_hat = P(Slice(8,11), Slice(0,Hp));
	
	x(Slice(),0) = x0;
	SX all_u = horzcat(u0, u);
	SX delta_u = all_u(Slice(), Slice(1,Hp+1)) - all_u(Slice(), Slice(0,Hp));
	
	// Optimization variables
	SX w = u(Slice(),0);
	for (int i=1; i<Hp; i++) {
		w = vertcat(w, u(Slice(),i));
	}
	vector<double> lbw (6*Hp);
	vector<double> ubw (6*Hp);
	for (int i=0; i<6*Hp; i++) {
		lbw[i] = -ubound;
		ubw[i] = +ubound;
	}
	
	// Other variables of the opt problem
	SX obj_fun = 0.0;
	SX g;
	vector<double> ubg;
	vector<double> lbg;
	
	// Weights (adaptive) calculation: direction from actual to desired position
	SX Qa;
	if (opt_Qa == true) {
	  Qa = get_adaptive_Q(Rp, x0, COM_coord_lc);
  }
  else {
  	Qa = 1;
  }

	
	// Optimization loop
	for (int k=0; k<Hp; k++) {

		// Model Dynamics Constraint -> Definition
		x(Slice(),k+1) = SX::mtimes(A_DM,x(Slice(),k))
									 + SX::mtimes(B_DM,u(Slice(),k))
	                 + COM.dt*SX::mtimes(f_DM,1);
		
		
		// Constraint: Constant distance between upper corners
		SX x_ctrl = SX::sym("x_ctrl", COM.coord_ctrl.size());
		for (int i=0; i<COM.coord_ctrl.size(); i++) {
		  x_ctrl(i) = x(COM.coord_ctrl(i),k+1);
		}
		g = vertcat(g, pow(x_ctrl(1) - x_ctrl(0), 2) + pow(x_ctrl(3) - x_ctrl(2), 2) +
		               pow(x_ctrl(5) - x_ctrl(4), 2) - pow(lCloth, 2) );

		lbg.insert(lbg.end(), 1, -gbound);
		ubg.insert(ubg.end(), 1, +gbound);
		
		// Objective function
		SX x_err = SX::sym("x_err", COM_coord_lc.size());
		for (int i=0; i<COM_coord_lc.size(); i++) {
			x_err(i) = x(COM_coord_lc(i),k+1) - Rp(i,k+1);
		}
		obj_fun += W_Q*SX::mtimes(SX::mtimes(x_err.T(), Qa), x_err);
		if (opt_du == false) {
			obj_fun += W_R*SX::mtimes(u(Slice(),k).T(), u(Slice(),k));
		}
		else {
			obj_fun += W_R*SX::mtimes(delta_u(Slice(),k).T(), delta_u(Slice(),k));
		}
	}

	// Encapsulate in controller object
	SXDict nlp_prob = {{"f",obj_fun},{"x",w},{"g",g},{"p",P}};
	Dict nlp_opts=Dict();
	nlp_opts["ipopt.print_level"] = 0;
	nlp_opts["ipopt.sb"] = "yes";
	nlp_opts["print_time"] = 0;
	Function controller = nlpsol("ctrl_sol", "ipopt", nlp_prob, nlp_opts);
	
	
	
	// -----------------------------------
	// 3. EXECUTION OF THE SIMULATION LOOP
	// -----------------------------------
	
	// Initial controls
	Eigen::VectorXd u_ini(6);
	Eigen::VectorXd u_bef(6);
	Eigen::VectorXd u_SOM(6);
	
	for (int i=0; i<SOM.coord_ctrl.size(); i++) {
		u_ini(i) = x_ini_SOM(SOM.coord_ctrl(i));
	}
	u_bef = u_ini;
	u_SOM = u_ini;
	
	// Get cloth orientation (rotation matrix)
	Eigen::Vector3d cloth_x(u_SOM(1)-u_SOM(0),
	                        u_SOM(3)-u_SOM(2),
	                        u_SOM(5)-u_SOM(4));
	Eigen::Vector3d cloth_y(-cloth_x(1), cloth_x(0), 0);
	Eigen::Vector3d cloth_z = cloth_x.cross(cloth_y);
	
	cloth_x = cloth_x/cloth_x.norm();
	cloth_y = cloth_y/cloth_y.norm();
	cloth_z = cloth_z/cloth_z.norm();
	
	Eigen::Matrix3d Rcloth;
	Eigen::Matrix3d Rtcp;
	Rcloth << cloth_x, cloth_y,  cloth_z;
	Rtcp   << cloth_y, cloth_x, -cloth_z;
	
	// Auxiliary variables for base changes
	Eigen::VectorXd phi_red(3*COMlength);
  Eigen::VectorXd dphi_red(3*COMlength);
	Eigen::MatrixXd pos_ini_COM(COMlength,3);
	Eigen::MatrixXd vel_ini_COM(COMlength,3);
	Eigen::MatrixXd pos_ini_COM_rot(COMlength,3);
	Eigen::MatrixXd vel_ini_COM_rot(COMlength,3);
	Eigen::VectorXd x_ini_COM_rot(2*3*COMlength);
	Eigen::MatrixXd pos_ini_SOM(SOMlength,3);
	Eigen::MatrixXd vel_ini_SOM(SOMlength,3);
	Eigen::MatrixXd pos_ini_SOM_rot(SOMlength,3);
	Eigen::MatrixXd vel_ini_SOM_rot(SOMlength,3);
	Eigen::VectorXd state_SOM_rot(2*3*SOMlength);
	Eigen::MatrixXd pos_nxt_SOM(SOMlength,3);
	Eigen::MatrixXd vel_nxt_SOM(SOMlength,3);
	Eigen::MatrixXd pos_nxt_SOM_rot(SOMlength,3);
	Eigen::MatrixXd vel_nxt_SOM_rot(SOMlength,3);
  Eigen::MatrixXd Traj_l_Hp_rot(Hp,3);
  Eigen::MatrixXd Traj_r_Hp_rot(Hp,3);
  Eigen::VectorXd u_rot = Eigen::VectorXd::Zero(6);
  Eigen::MatrixXd u_rot2(3,2);
	Eigen::MatrixXd u_lin2(3,2);
	Eigen::VectorXd u_lin(6);
	
	// Initialize input parameters
	Eigen::MatrixXd in_params(2+6+3, max(n_states, Hp+1));
	
	// Reference trajectory in horizon
	Eigen::MatrixXd Traj_l_Hp(Hp+1,3);
	Eigen::MatrixXd Traj_r_Hp(Hp+1,3);
	
	// Initialize SOM state
	state_SOM.resize(6*SOMlength);
	state_SOM = x_ini_SOM;
	
	
	// START SIMULATION LOOP
	// Matlab code has Hp and sHp as moving window start-end
	// Changed to time instant (tk) and fixed Hp so window is tk : tk+Hp
	int tk = 0;
	
	// NOT REAL TIME! Can be changed to 1/Ts but optimizer is blocking
	ros::Rate rate(20);
	while(rosnh.ok()) {
	
		// Save initial iteration time
		ros::Time iterT0 = ros::Time::now();
	
		// Check subscriptions
		ros::spinOnce();
	
		// Slice reference (constant in the window near the end)
	  if(tk >= phi_l_Traj.rows()-(Hp+1)) {
	    Traj_l_Hp = Eigen::VectorXd::Ones(Hp+1)*phi_l_Traj.bottomRows(1);
	    Traj_r_Hp = Eigen::VectorXd::Ones(Hp+1)*phi_r_Traj.bottomRows(1);
	  }
	  else{
	    Traj_l_Hp = phi_l_Traj.block(tk,0, Hp+1,3);
	    Traj_r_Hp = phi_r_Traj.block(tk,0, Hp+1,3);
	  }
	  //ROS_WARN_STREAM(endl<<Traj_l_Hp<<endl<<endl<<Traj_r_Hp<<endl);
	  
		// Get reduced states (SOM->COM)
    tie(phi_red, dphi_red) = take_reduced_mesh(state_SOM.segment(0,3*SOMlength),
  																						 state_SOM.segment(3*SOMlength,3*SOMlength),
  																						 nxS, nxC);
  	
  	// Update COM state																					 
  	x_ini_COM << phi_red, dphi_red;
	  
		// Rotate initial position to cloth base
		pos_ini_COM.col(0) = x_ini_COM.segment(0, COMlength);
		pos_ini_COM.col(1) = x_ini_COM.segment(1*COMlength, COMlength);
		pos_ini_COM.col(2) = x_ini_COM.segment(2*COMlength, COMlength);
		vel_ini_COM.col(0) = x_ini_COM.segment(3*COMlength, COMlength);
		vel_ini_COM.col(1) = x_ini_COM.segment(4*COMlength, COMlength);
		vel_ini_COM.col(2) = x_ini_COM.segment(5*COMlength, COMlength);
		
		pos_ini_COM_rot = (Rcloth.inverse() * pos_ini_COM.transpose()).transpose();
		vel_ini_COM_rot = (Rcloth.inverse() * vel_ini_COM.transpose()).transpose();
		x_ini_COM_rot << pos_ini_COM_rot.col(0),
		                 pos_ini_COM_rot.col(1),
		                 pos_ini_COM_rot.col(2),
		                 vel_ini_COM_rot.col(0),
		                 vel_ini_COM_rot.col(1),
		                 vel_ini_COM_rot.col(2);
		
		// Inverse = transpose, can be rearranged
		Traj_l_Hp_rot = (Rcloth.inverse() * Traj_l_Hp.transpose()).transpose();
		Traj_r_Hp_rot = (Rcloth.inverse() * Traj_r_Hp.transpose()).transpose();
		
	
		// Reset reference
		in_params.setZero(in_params.rows(), in_params.cols());
	
		in_params.block(0,0, 1,n_states) = x_ini_COM_rot.transpose();
		in_params.block(1,0, 1,6) 	 = u_rot.transpose();
		in_params.block(2,0, 1,Hp+1) = Traj_l_Hp_rot.col(0).transpose();
		in_params.block(3,0, 1,Hp+1) = Traj_r_Hp_rot.col(0).transpose();
		in_params.block(4,0, 1,Hp+1) = Traj_l_Hp_rot.col(1).transpose();
		in_params.block(5,0, 1,Hp+1) = Traj_r_Hp_rot.col(1).transpose();	
		in_params.block(6,0, 1,Hp+1) = Traj_l_Hp_rot.col(2).transpose();
		in_params.block(7,0, 1,Hp+1) = Traj_r_Hp_rot.col(2).transpose();
		//in_params.block(8,0, 3,Hp) = ... //d_hat
	
		// Initial seed of the optimization
		Eigen::MatrixXd dRef = in_params.block(2,1, 6,Hp) - in_params.block(2,0, 6,Hp);
		Eigen::Map<Eigen::VectorXd> args_x0(dRef.data(), dRef.size());
	
		// Transform variables for solver
		DM x0_dm = DM::zeros(6*Hp, 1);
		memcpy(x0_dm.ptr(), args_x0.data(), sizeof(double)*6*Hp);
	
		DM p_dm = DM::zeros(in_params.rows(), in_params.cols());
		memcpy(p_dm.ptr(), in_params.data(), sizeof(double)*in_params.rows()*in_params.cols());
	
		// Create the structure of parameters
		map<string, DM> arg, sol;
		arg["lbx"] = lbw;
		arg["ubx"] = ubw;
		arg["lbg"] = lbg;
		arg["ubg"] = ubg;
		arg["x0"]  = x0_dm;
		arg["p"]   = p_dm;
	
		// Find the solution
		sol = controller(arg);
		DM wsol = sol["x"];
		
		// Check how long it took
		ros::Duration optiDT = ros::Time::now() - iterT0;
		double optiDTms = optiDT.toSec() * 1000;
		int optiSteps = ceil((optiDTms/1000)/Ts);
		ROS_WARN_STREAM("Opt.time: "<<optiDTms<<" ms \t("<<optiSteps<<" steps)");
		
	
		// Get control actions from the solution
		//  They are upper corner displacements (incremental pos)
		//  And they are in local cloth base (rot)
		DM usol = DM::zeros(Hp,6);
		for (int i=0; i<6*Hp; i++) {
			usol(i/6,i%6) = wsol(i);
		}
		DM u_rot_dm = usol(0,Slice()); //u_rot1
		vector<double> u_rotv = (vector<double>) u_rot_dm;
	  u_rot = Eigen::VectorXd::Map(u_rotv.data(), u_rotv.size());
	  
		// Convert back to global base
		u_rot2.col(0) << u_rot(0), u_rot(2), u_rot(4);
		u_rot2.col(1) << u_rot(1), u_rot(3), u_rot(5);
		u_lin2 = Rcloth * u_rot2;
		u_lin << u_lin2.row(0).transpose(), u_lin2.row(1).transpose(), u_lin2.row(2).transpose();
		
		// Output (input of Cartesian ctrl) is u_SOM
		u_SOM = u_lin + u_bef;
		u_bef = u_SOM;
		
		// Update uSOM history (6xNH)
		uSOMhist.col(uhistID) = u_SOM;
		uhistID = (uhistID+1)%MaxVSteps;
		
		
		// Linear SOM uses local base variables (rot)
		pos_ini_SOM.col(0) = state_SOM.segment(0, SOMlength);
		pos_ini_SOM.col(1) = state_SOM.segment(1*SOMlength, SOMlength);
		pos_ini_SOM.col(2) = state_SOM.segment(2*SOMlength, SOMlength);
		vel_ini_SOM.col(0) = state_SOM.segment(3*SOMlength, SOMlength);
		vel_ini_SOM.col(1) = state_SOM.segment(4*SOMlength, SOMlength);
		vel_ini_SOM.col(2) = state_SOM.segment(5*SOMlength, SOMlength);
		
		pos_ini_SOM_rot = (Rcloth.inverse() * pos_ini_SOM.transpose()).transpose();
		vel_ini_SOM_rot = (Rcloth.inverse() * vel_ini_SOM.transpose()).transpose();
		state_SOM_rot << pos_ini_SOM_rot.col(0),
		                 pos_ini_SOM_rot.col(1),
		                 pos_ini_SOM_rot.col(2),
		                 vel_ini_SOM_rot.col(0),
		                 vel_ini_SOM_rot.col(1),
		                 vel_ini_SOM_rot.col(2);
	
		// Simulate a step
	  state_SOM_rot = A_SOM*state_SOM_rot + B_SOM*u_rot + Ts*f_SOM;
	  
	  // Convert back to global axes
	  pos_nxt_SOM_rot.col(0) = state_SOM_rot.segment(0, SOMlength);
		pos_nxt_SOM_rot.col(1) = state_SOM_rot.segment(1*SOMlength, SOMlength);
		pos_nxt_SOM_rot.col(2) = state_SOM_rot.segment(2*SOMlength, SOMlength);
		vel_nxt_SOM_rot.col(0) = state_SOM_rot.segment(3*SOMlength, SOMlength);
		vel_nxt_SOM_rot.col(1) = state_SOM_rot.segment(4*SOMlength, SOMlength);
		vel_nxt_SOM_rot.col(2) = state_SOM_rot.segment(5*SOMlength, SOMlength);
		
		pos_nxt_SOM = (Rcloth * pos_nxt_SOM_rot.transpose()).transpose();
		vel_nxt_SOM = (Rcloth * vel_nxt_SOM_rot.transpose()).transpose();
		state_SOM << pos_nxt_SOM.col(0),
		             pos_nxt_SOM.col(1),
		             pos_nxt_SOM.col(2),
		             vel_nxt_SOM.col(0),
		             vel_nxt_SOM.col(1),
		             vel_nxt_SOM.col(2);
  	
  	// Get new Cloth orientation (rotation matrix)
  	cloth_x << u_SOM(1)-u_SOM(0),
	             u_SOM(3)-u_SOM(2),
	             u_SOM(5)-u_SOM(4);
		cloth_y << -cloth_x(1), cloth_x(0), 0;
		cloth_z = cloth_x.cross(cloth_y);
	
		cloth_x = cloth_x/cloth_x.norm();
		cloth_y = cloth_y/cloth_y.norm();
		cloth_z = cloth_z/cloth_z.norm();
	
		Rcloth << cloth_x, cloth_y,  cloth_z;
		Rtcp   << cloth_y, cloth_x, -cloth_z;
  	
  	//ROS_WARN_STREAM(endl<<Rtcp<<endl);
  	
  	// Transform orientation to quaternion
  	tf2::Matrix3x3 tfRtcp;
		tfRtcp.setValue(Rtcp(0,0), Rtcp(0,1), Rtcp(0,2),
		              	Rtcp(1,0), Rtcp(1,1), Rtcp(1,2),
		              	Rtcp(2,0), Rtcp(2,1), Rtcp(2,2));
		
		tf2::Quaternion tfQtcp;
		tfRtcp.getRotation(tfQtcp);
		geometry_msgs::Quaternion Qtcp = tf2::toMsg(tfQtcp);
  	
		
		// Publish control action: Two corners
		mpc_pkg::TwoPoints u_SOM_pub;
		
		u_SOM_pub.pt1.x = u_SOM(0);
		u_SOM_pub.pt2.x = u_SOM(1);
		u_SOM_pub.pt1.y = u_SOM(2);
		u_SOM_pub.pt2.y = u_SOM(3);
		u_SOM_pub.pt1.z = u_SOM(4);
		u_SOM_pub.pt2.z = u_SOM(5);
		
		pub_usom.publish(u_SOM_pub);
		
		// Cloth-TCP Base offset in Robot coords
		Eigen::Vector3d tcp_offset = Rcloth * tcp_offset_local;
		
		// Publish control action (TCP) if not NaN
		if (!u_SOM.hasNaN()) {
		  cartesian_msgs::CartesianCommand u_WAM_pub;
		  geometry_msgs::PoseStamped u_TCP_pub;
		
		  u_TCP_pub.header.stamp = ros::Time::now();
		  u_TCP_pub.header.frame_id = "iri_wam_link_base";
		  u_TCP_pub.pose.position.x = (u_SOM(0)+u_SOM(1))/2 + tcp_offset(0);
		  u_TCP_pub.pose.position.y = (u_SOM(2)+u_SOM(3))/2 + tcp_offset(1);
		  u_TCP_pub.pose.position.z = (u_SOM(4)+u_SOM(5))/2 + tcp_offset(2);
		  u_TCP_pub.pose.orientation = Qtcp;
		
		  u_WAM_pub.desired_pose = u_TCP_pub;
		  u_WAM_pub.duration = 0.010;
		
		  pub_utcp.publish(u_TCP_pub);
		  pub_uwam.publish(u_WAM_pub);
		}
		
		// Debugging
		//ROS_WARN_STREAM(endl<<u_SOM<<endl);
		
		
		// Increase counter
		tk++;
		
		// End after trajectory is completed
		if (tk > phi_l_Traj.rows() + 4/Ts) {
		  break;
		}
		
		// Execute at a fixed rate
		rate.sleep();
	
	}
	// END LOOP
	
	
  ROS_INFO_STREAM("Reached the end!" << endl);
  
}































