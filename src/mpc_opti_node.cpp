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
#include <mpc_pkg/OptiData.h>
#include <mpc_pkg/HorizonControls.h>
#include <std_srvs/Empty.h>


using namespace std;
using namespace casadi;


// GLOBAL VARIABLES
string datapath = "/home/robot/Desktop/ALuque/mpc_ros/src/mpc_node/data/";
int NTraj = 0;
int nCOM = 0;
int Hp = 0;
double Ts = 0.00;
bool shutdown_flag = false;
Eigen::MatrixXd in_params;
Eigen::Matrix3d Rcloth;
Eigen::VectorXd u_bef(6);
// ----------------

// MAIN CONTROL PARAMETERS
// --------------------------
// Objective Function Weights
double W_Q = 1.0;
double W_R = 0.2;
// Constraint bounds
double ubound = 0.01;
double gbound = 0.0;
// Structure variants
bool opt_du = true;
bool opt_Qa = false;
// --------------------------


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
void inidataReceived(const mpc_pkg::OptiData& msg) {
	
	// Data is:
	// - Vectors of "in_params", Traj_[]_Hp_rot, u_rot and x_ini_COM_rot, updated
	// - Rcloth, updated to the input data so new u_rot can be changed back
	// - u_bef, to add to the incremental u_lin and get absolute positions
	
	in_params.setZero(in_params.rows(), in_params.cols());

	in_params.row(0) = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>
                     (((vector<double>) msg.xinicom_rot).data(), 6*nCOM*nCOM).transpose();
  in_params.block(1,0, 1,6) 	 = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>
  															 (((vector<double>) msg.u_rot).data(), 6).transpose();
	in_params.block(2,0, 1,Hp+1) = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>
                                 (((vector<double>) msg.traj_left_x).data(),  Hp+1).transpose();
	in_params.block(3,0, 1,Hp+1) = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>
                                 (((vector<double>) msg.traj_right_x).data(), Hp+1).transpose();
	in_params.block(4,0, 1,Hp+1) = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>
                                 (((vector<double>) msg.traj_left_y).data(),  Hp+1).transpose();
	in_params.block(5,0, 1,Hp+1) = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>
                                 (((vector<double>) msg.traj_right_y).data(), Hp+1).transpose();
	in_params.block(6,0, 1,Hp+1) = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>
                                 (((vector<double>) msg.traj_left_z).data(),  Hp+1).transpose();
	in_params.block(7,0, 1,Hp+1) = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>
                                 (((vector<double>) msg.traj_right_z).data(), Hp+1).transpose();
	//in_params.block(8,0, 3,Hp) = ... //d_hat
	
  Rcloth.col(0) = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>
                  (((vector<double>) msg.Rcloth_x).data(), 3);
  Rcloth.col(1) = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>
                  (((vector<double>) msg.Rcloth_y).data(), 3);
  Rcloth.col(2) = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>
                  (((vector<double>) msg.Rcloth_z).data(), 3);
  
  u_bef = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>
          (((vector<double>) msg.u_bef).data(), 6);
	
}


// ROS SERVICES - CALLBACK FUNCTIONS
bool shutdownCallback(std_srvs::Empty::Request &req, std_srvs::Empty::Response &resp) {
	shutdown_flag = true;
	ROS_INFO_STREAM("Shutting down Optimizer");
	return true;

}




// ---------------------------
// -------- MAIN PROG --------
// ---------------------------

int main(int argc, char **argv) {

	// Initialize the ROS system and become a node.
  ros::init(argc, argv, "mpc_opti_node");
  ros::NodeHandle rosnh;
  
	// Define client & server objects to all services
	ros::ServiceServer srv_shutdown = rosnh.advertiseService("node_shutdown", &shutdownCallback);
	
  //ros::ServiceClient clt_foo = rosnh.serviceClient<node::Service>("service_name");
  //ros::service::waitForService("service_name");
  
  // Define Publishers
  ros::Publisher pub_usomhp = rosnh.advertise<mpc_pkg::HorizonControls>
                              ("mpc_controller/u_som_hp", 1000);
  
  // Define Subscribers
  ros::Subscriber sub_inidata = rosnh.subscribe("mpc_controller/opti_inidata",
                                                 1000, &inidataReceived);
  
  // Get parameters from launch
  if(!rosnh.getParam("/mpc_opti_node/datapath", datapath)) {
  	ROS_ERROR("Need to define the datapath (where reference csv files are) parameter.");
  }
  if(!rosnh.getParam("/mpc_opti_node/NTraj", NTraj)) {
  	ROS_ERROR("Need to define the NTraj (reference trajectory number) parameter.");
  }
  if(!rosnh.getParam("/mpc_opti_node/nCOM", nCOM)) {
  	ROS_ERROR("Need to define the nCOM (COM mesh side size) parameter.");
  }
  if(!rosnh.getParam("/mpc_opti_node/Hp", Hp)) {
  	ROS_ERROR("Need to define the Hp (prediction horizon) parameter.");
  }
  if(!rosnh.getParam("/mpc_opti_node/Ts", Ts)) {
  	ROS_ERROR("Need to define the Ts (sample time) parameter.");
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
  
  
  // Initialization of COM variable
  LinMdl COM;
  
  // Load parameter table and get row for COM
  // LUT cols: MdlSz, Ts, theta[7]
  Eigen::MatrixXd thetaLUT = getCSVcontent(datapath+"ThetaMdl_LUT.csv");
  bool COMparams = false;
  for (int LUTi=0; LUTi<thetaLUT.rows(); LUTi++) {
		if (thetaLUT(LUTi, 1) != Ts) continue;
		
		Eigen::MatrixXd theta_i = thetaLUT.block(LUTi,2, 1,7);
		
		// Set model parameters
		if (thetaLUT(LUTi, 0) == nCOM) {
			if (COMparams) ROS_WARN_STREAM("Several rows match the same COM parameters!");
			
		  COM.stiffness << theta_i(0), theta_i(1), theta_i(2);
  		COM.damping   << theta_i(3), theta_i(4), theta_i(5);
  		COM.z_sum      = theta_i(6);
  		COMparams = true;
		}
		
  }
  // If no row matched the settings
  if (!COMparams) {
  	ROS_WARN_STREAM("No rows match the COM parameters! Setting theta to default.");
  	
  	nCOM = 4;
		COM.stiffness << -200.0, -15.0, -200.0;
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
	
  // Define initial position of the nodes (for ext_force)
  // Second half of the vector is velocities (initial = 0)
  Eigen::MatrixXd posCOM(COMlength,3);
  posCOM = create_lin_mesh(lCloth, nCOM, cCloth, aCloth);
  
  Eigen::VectorXd x_ini_COM(2*3*COMlength);
  x_ini_COM.setZero(2*3*COMlength);
  
  x_ini_COM.segment(0, COMlength) = posCOM.col(0);
  x_ini_COM.segment(1*COMlength, COMlength) = posCOM.col(1);
  x_ini_COM.segment(2*COMlength, COMlength) = posCOM.col(2);
  
  // Rotate initial COM positions to XZ plane
  Eigen::Matrix3d RCloth_ini;
  RCloth_ini << cos(aCloth), -sin(aCloth), 0,
  				      sin(aCloth),  cos(aCloth), 0,
  				               0,             0, 1;               
	Eigen::MatrixXd posCOM_XZ(COMlength,3);
	posCOM_XZ = (RCloth_ini.inverse() * posCOM.transpose()).transpose();
  

 	// Get the linear model for the COM
 	Eigen::MatrixXd A(6*COMlength,6*COMlength);
 	Eigen::MatrixXd B(6*COMlength,6);
 	Eigen::VectorXd ext_force(6*COMlength);
 	A.setZero(6*COMlength,6*COMlength);
 	B.setZero(6*COMlength,6);
 	ext_force.setZero(6*COMlength);
 	
	tie(COM, A, B, ext_force) = init_linear_model(COM, posCOM_XZ);
	
  
  
	// ----------------------------------
	// 2. OPTIMIZATION PROBLEM DEFINITION
	// ----------------------------------

	// Solver timeout: less than prediction time
	double timeout_s = Ts*Hp/4;
	
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
	nlp_opts["ipopt.max_cpu_time"] = timeout_s;
	nlp_opts["ipopt.sb"] = "yes";
	nlp_opts["print_time"] = 0;
	Function controller = nlpsol("ctrl_sol", "ipopt", nlp_prob, nlp_opts);
	
	
	
	// ----------------------------------
	// 3. EXECUTION OF THE OPTIMIZER LOOP
	// ----------------------------------
	
	// Initial controls
	Eigen::VectorXd u_SOM(6);
	
	// Auxiliary variables for base changes and storage
	vector<double> u1_rotv;
	vector<double> u2_rotv;
	vector<double> u3_rotv;
	vector<double> u4_rotv;
	vector<double> u5_rotv;
	vector<double> u6_rotv;
  Eigen::VectorXd u1_rot;
  Eigen::VectorXd u2_rot;
  Eigen::VectorXd u3_rot;
  Eigen::VectorXd u4_rot;
  Eigen::VectorXd u5_rot;
  Eigen::VectorXd u6_rot;
	
	Eigen::MatrixXd uHp_rot(Hp,6);
	Eigen::MatrixXd uHp_p1_rot2(3,Hp);
	Eigen::MatrixXd uHp_p2_rot2(3,Hp);
	Eigen::MatrixXd uHp_p1_lin2(3,Hp);
	Eigen::MatrixXd uHp_p2_lin2(3,Hp);
	Eigen::MatrixXd uHp_lin(Hp,6);
	Eigen::MatrixXd uHp_SOM(Hp,6);
	
	
	// Resize input parameters matrix for all states
	in_params.resize(2+6+3, max(n_states, Hp+1));

	
	// Wait for initial optidata
	ROS_INFO_STREAM("Initialized Optimizer");
	boost::shared_ptr<mpc_pkg::OptiData const> OptiData0;
	OptiData0 = ros::topic::waitForMessage<mpc_pkg::OptiData>("/mpc_controller/opti_inidata");
  
  
	// START LOOP
	ros::Rate rate(1/Ts);
	while(rosnh.ok() && !shutdown_flag) {
	
		// Save initial iteration time
		ros::Time iterT0 = ros::Time::now();
	
		// Check subscriptions (in_params, Rcloth, u_bef)
		ros::spinOnce();
	
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
		
		
		// Check how long it took
		ros::Duration optiDT = ros::Time::now() - iterT0;
		//int optiSteps = ceil(optiDT.toSec()/Ts);
		//ROS_INFO_STREAM("Opt.time: "<<1000*optiDT.toSec()<<" ms \t("<<optiSteps<<" steps)");
		if (optiDT.toSec() >= timeout_s) {
			int optiSteps = ceil(optiDT.toSec()/Ts);
		  ROS_WARN_STREAM("SOLVER TIMED OUT ("<<
		                  1000*optiDT.toSec() <<" ms / "<<
		                  optiSteps<<" steps)");
		  continue;
		}
		

		// Get control actions from the solution
		//  They are upper corner displacements (incremental pos)
		//  And they are in local cloth base (rot)
		DM wsol = sol["x"];
		DM usol = DM::zeros(Hp,6);
		for (int i=0; i<6*Hp; i++) {
			usol(i/6,i%6) = wsol(i);
		}
		
		// Process controls for the whole horizon
		u1_rotv = (vector<double>) usol(Slice(),0); //x1
		u2_rotv = (vector<double>) usol(Slice(),1); //x2
		u3_rotv = (vector<double>) usol(Slice(),2); //y1
		u4_rotv = (vector<double>) usol(Slice(),3); //y2
		u5_rotv = (vector<double>) usol(Slice(),4); //z1
		u6_rotv = (vector<double>) usol(Slice(),5); //z2
	  u1_rot = Eigen::VectorXd::Map(u1_rotv.data(), u1_rotv.size());
	  u2_rot = Eigen::VectorXd::Map(u2_rotv.data(), u2_rotv.size());
	  u3_rot = Eigen::VectorXd::Map(u3_rotv.data(), u3_rotv.size());
	  u4_rot = Eigen::VectorXd::Map(u4_rotv.data(), u4_rotv.size());
	  u5_rot = Eigen::VectorXd::Map(u5_rotv.data(), u5_rotv.size());
	  u6_rot = Eigen::VectorXd::Map(u6_rotv.data(), u6_rotv.size());
	  uHp_rot << u1_rot, u2_rot, u3_rot, u4_rot, u5_rot, u6_rot;
	  
	  uHp_p1_rot2.row(0) = u1_rot.transpose();
	  uHp_p1_rot2.row(1) = u3_rot.transpose();
	  uHp_p1_rot2.row(2) = u5_rot.transpose();
	  uHp_p2_rot2.row(0) = u2_rot.transpose();
	  uHp_p2_rot2.row(1) = u4_rot.transpose();
	  uHp_p2_rot2.row(2) = u6_rot.transpose();
	  
	  uHp_p1_lin2 = Rcloth * uHp_p1_rot2;
	  uHp_p2_lin2 = Rcloth * uHp_p2_rot2;
	  
	  uHp_lin.col(0) = uHp_p1_lin2.row(0).transpose();
	  uHp_lin.col(1) = uHp_p2_lin2.row(0).transpose();
	  uHp_lin.col(2) = uHp_p1_lin2.row(1).transpose();
	  uHp_lin.col(3) = uHp_p2_lin2.row(1).transpose();
	  uHp_lin.col(4) = uHp_p1_lin2.row(2).transpose();
	  uHp_lin.col(5) = uHp_p2_lin2.row(2).transpose();
	  
	  // u_SOM(n) = u_bef + Sum(u_lin(i))_(i=0 to n-1)
	  uHp_SOM.row(0) = uHp_lin.row(0) + u_bef.transpose();
	  for (int i=1; i<Hp; i++) {
	  	uHp_SOM.row(i) = uHp_lin.row(i) + uHp_SOM.row(i-1);
	  }
	  
		// Publish control actions
		mpc_pkg::HorizonControls uHp_SOM_pub;
		uHp_SOM_pub.u1Hp = vector<double>(uHp_SOM.col(0).data(),
								uHp_SOM.col(0).size()+uHp_SOM.col(0).data());
		uHp_SOM_pub.u2Hp = vector<double>(uHp_SOM.col(1).data(),
								uHp_SOM.col(1).size()+uHp_SOM.col(1).data());
		uHp_SOM_pub.u3Hp = vector<double>(uHp_SOM.col(2).data(),
								uHp_SOM.col(2).size()+uHp_SOM.col(2).data());
		uHp_SOM_pub.u4Hp = vector<double>(uHp_SOM.col(3).data(),
								uHp_SOM.col(3).size()+uHp_SOM.col(3).data());
		uHp_SOM_pub.u5Hp = vector<double>(uHp_SOM.col(4).data(),
								uHp_SOM.col(4).size()+uHp_SOM.col(4).data());
		uHp_SOM_pub.u6Hp = vector<double>(uHp_SOM.col(5).data(),
								uHp_SOM.col(5).size()+uHp_SOM.col(5).data());
	  
	  pub_usomhp.publish(uHp_SOM_pub);
		
		
		// Debugging
		//ROS_INFO_STREAM(endl<<u_SOM<<endl);
		
		// Execute at a fixed rate
		rate.sleep();
	
	}
	// END LOOP
	
  
}































