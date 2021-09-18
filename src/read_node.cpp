#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include "general_functions.h"
#include <ros/ros.h>
#include "cartesian_msgs/CartesianCommand.h"
#include "geometry_msgs/PoseArray.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Quaternion.h"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <mpc_pkg/ReferenceWindow.h>

using namespace std;

//pub /iri_wam_controller/cartesian_controller/goal iri_wam_common_msgs/DMPTrackerActionGoal '{header:{seq:0,stamp:{secs:0,nsecs:0},frame_id:''},goal_id:{stamp:{secs:0,nsecs:0},id:''},goal:{initial:{positions:[-1.5708, 0.1257, 0, 1.9300, 0, 1.0859, 0],velocities:[0],accelerations:[0],effort:[0],time_from_start:{secs:0,nsecs:0}},goal:{positions:[0],velocities:[0],accelerations:[0],effort:[0],time_from_start:{secs:0,nsecs:0}}}}' -1


// GLOBAL VARIABLES
string datapath = "/home/robot/iri-lab/labrobotica/drivers/irilibbarrett/bin/Trajs/TrajWAMs/";
string datafile = "TrajWAM_8";
// ----------------



// ---------------------------
// -------- MAIN PROG --------
// ---------------------------

int main(int argc, char **argv) {

	// Initialize the ROS system and become a node.
  ros::init(argc, argv, "read_node");
  ros::NodeHandle rosnh;
  
	// Define client objects to all services
  //ros::ServiceClient clt_foo = rosnh.serviceClient<node::Service>("service_name");
  //ros::service::waitForService("service_name");
  
  // Define Publishers
  ros::Publisher pub_pose = rosnh.advertise<geometry_msgs::PoseStamped>
                            ("mpc_controller/traj_tcp_pose", 1000);
  ros::Publisher pub_cart = rosnh.advertise<cartesian_msgs::CartesianCommand>
                            ("iri_wam_controller/CartesianControllerNewGoal", 1000);
  
  // Get parameters from launch
  if(!rosnh.getParam("/read_node/datapath", datapath)) {
  	ROS_ERROR("Need to define the datapath (where trajectory csv files are) parameter.");
  }
  if(!rosnh.getParam("/read_node/datafile", datafile)) {
  	ROS_ERROR("Need to define the datafile (trajectory csv file) parameter.");
  }
  
	
	// Load trajectories to follow
  Eigen::MatrixXd RawTraj = getCSVcontent(datapath+datafile+".csv");
  int TrajSize = RawTraj.rows();
  
  // Set rate (assumed constant)
  double Ts = RawTraj(1,0) - RawTraj(0,0);
  int traj_rate = 1/Ts;

	int tk=0;
	ros::Rate rate(traj_rate);
	while(rosnh.ok()) {
	
	  
	  Eigen::MatrixXd TrajLine = RawTraj.row(tk);

		// Create the poses and pose arrays
		cartesian_msgs::CartesianCommand command_pub;
		geometry_msgs::PoseStamped pose_pub;
		
		pose_pub.header.stamp = ros::Time::now();
		pose_pub.header.frame_id = "iri_wam_link_base";
	  
	  // Traj line: time, joints1-7, xyz, wxyz
		pose_pub.pose.position.x = RawTraj(tk, 8);
		pose_pub.pose.position.y = RawTraj(tk, 9);
		pose_pub.pose.position.z = RawTraj(tk, 10);
		pose_pub.pose.orientation.w = RawTraj(tk, 11);
    pose_pub.pose.orientation.x = RawTraj(tk, 12);
    pose_pub.pose.orientation.y = RawTraj(tk, 13);
    pose_pub.pose.orientation.z = RawTraj(tk, 14);
    
    command_pub.desired_pose = pose_pub;
    command_pub.duration = Ts;
		
		// Publish the trajectories
		pub_pose.publish(pose_pub);
		pub_cart.publish(command_pub);
		
		// Increase counter
		if (tk < TrajSize-1) {
  		tk++;
		}
		
		// Execute at a fixed rate
		rate.sleep();
	}
	
}








