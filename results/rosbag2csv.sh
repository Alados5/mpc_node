#!/bin/bash

FILE="session_2021_07_08/cl6_traj7_ts25_hp20_feedback_state_w20"

rostopic echo -b $FILE.bag -p /cloth_segmentation/cloth_mesh > $FILE"_mesh.csv"
rostopic echo -b $FILE.bag -p /iri_wam_controller/libbarrett_link_tcp > $FILE"_tcp.csv"
rostopic echo -b $FILE.bag -p /joint_states > $FILE"_joints.csv"
rostopic echo -b $FILE.bag -p /mpc_controller/camera_pose > $FILE"_campose.csv"
rostopic echo -b $FILE.bag -p /mpc_controller/state_SOM > $FILE"_somstate.csv"
rostopic echo -b $FILE.bag -p /mpc_controller/u_TCP > $FILE"_utcp.csv"
