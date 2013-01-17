/*
 * Main.cpp
 *
 *  Created on: 17.01.2013
 *      Author: nico
 */

#include <ros/ros.h>
#include "Segmentation.hpp"
#include "Shrinker.hpp"

namespace ovenRecognizer {
ros::Publisher pub;
Segmentation seg;
Shrinker shr;

void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& cloud) {
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmpCloud1(
			new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmpCloud2(
			new pcl::PointCloud<pcl::PointXYZRGB>);
	const sensor_msgs::PointCloud2::Ptr outPutCloud(
			new sensor_msgs::PointCloud2);
	pcl::fromROSMsg(*cloud, *tmpCloud1);

	shr.resizeTo(tmpCloud1, tmpCloud2, 0.9, 1.4);
	seg.segmentFlat(tmpCloud2, tmpCloud1);

	seg.segmentCircle(tmpCloud1, tmpCloud2);

	pcl::toROSMsg(*tmpCloud2, *outPutCloud);

	pub.publish(outPutCloud);

}
}

int main(int argc, char** argv) {
	// Initialize ROS
	ros::init(argc, argv, "piripiri_shrinker");
	ros::NodeHandle nh;

	// Create a ROS subscriber for the input point cloud
	ros::Subscriber sub = nh.subscribe("/kinect_head/depth_registered/points",
			1, ovenRecognizer::cloud_cb);

	// Create a ROS publisher for the output point cloud
	ovenRecognizer::pub = nh.advertise<sensor_msgs::PointCloud2>(
			"/piripiri/depth_registered/points", 1);

	ros::spin();

	return 0;
}

