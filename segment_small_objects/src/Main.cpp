/*
 * Main.cpp
 *
 *  Created on: 17.01.2013
 *      Author: nico
 */

#include <ros/ros.h>
#include "Segmentation.hpp"
#include "Shrinker.hpp"
#include "Downsampler.hpp"
#include <pcl/common/centroid.h>

#include <math.h>

typedef pcl::PointXYZRGB PointT;
namespace ovenRecognizer {
ros::Publisher pub;
Segmentation<PointT> seg;
Shrinker<PointT> shr;
Downsampler<PointT> dwn;

void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& cloud) {

	pcl::PointCloud<PointT>::Ptr inputCloud(new pcl::PointCloud<PointT>);
	const sensor_msgs::PointCloud2::Ptr outPutCloud(
			new sensor_msgs::PointCloud2);

	pcl::fromROSMsg(*cloud, *inputCloud);

	dwn.downsample(inputCloud, 0.006f);

	shr.resizeTo(dwn.outputCloud, 0, 1.4);

	//shr.resizeTo(inputCloud, 0 , 1.4);

	seg.getEverythingOnTopOfTable(shr.outputCloud);

	seg.extractBigObjects(seg.outputCloud);

	seg.extractColors(seg.outputCloud);

	pcl::toROSMsg(*seg.outputCloud, *outPutCloud);

	pub.publish(outPutCloud);

	ROS_INFO("Size: %d", outPutCloud->width);

	ROS_INFO("done");

}
}

int main(int argc, char** argv) {
	// Initialize ROS
	ros::init(argc, argv, "oven_recognizer");
	ros::NodeHandle nh;

	// Create a ROS subscriber for the input point cloud
	ros::Subscriber sub = nh.subscribe("/camera/depth_registered/points",
			1, ovenRecognizer::cloud_cb);

	// Create a ROS publisher for the output point cloud
	ovenRecognizer::pub = nh.advertise<sensor_msgs::PointCloud2>(
			"/piripiri/depth_registered/points", 1);
	ros::Rate r(100);
	ros::spin();
	r.sleep();
	return 0;
}

