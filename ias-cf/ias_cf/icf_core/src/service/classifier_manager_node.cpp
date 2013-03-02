/*
 * main.cpp
 *
 *  Created on: Mar 1, 2011
 *      Author: ferka
 */
#include "icf_core/service/ClassifierManager.h"
#include "ros/ros.h"

int main(int argc, char **argv)
{
  ROS_INFO("Starting service");
  ros::init(argc, argv, "ias_classifier_manager"); //name of node
  ros::NodeHandle n("~");
  new icf::ClassifierManager(n);
  ros::MultiThreadedSpinner spinner(8);
  spinner.spin();
  return 0;
}
