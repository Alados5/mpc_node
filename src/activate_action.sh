#!/bin/bash

rostopic pub /iri_wam_controller/cartesian_controller/goal iri_wam_common_msgs/DMPTrackerActionGoal "{goal: {initial: {positions: [-1.5675, 0.1294, 0.0013, 2.0328, -0.0175, 1.0006, 0.0012]}}}" -1



