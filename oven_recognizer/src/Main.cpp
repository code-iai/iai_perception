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
#include "Center.hpp"
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

typedef pcl::PointXYZRGB PointT;
namespace ovenRecognizer {
ros::Publisher pub;
Segmentation<PointT> seg;
Shrinker<PointT> shr;
Downsampler<PointT> dwn;
Center<PointT> cen;

void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& cloud) {
	pcl::PointCloud<PointT>::Ptr inputCloud(new pcl::PointCloud<PointT>);
	const sensor_msgs::PointCloud2::Ptr outPutCloud(
			new sensor_msgs::PointCloud2);

	pcl::fromROSMsg(*cloud, *inputCloud);
	ros::Time startTime = ros::Time::now();

	tf::TransformBroadcaster br;
	tf::StampedTransform transform;

	dwn.downsample(inputCloud);

	shr.resizeTo(dwn.outputCloud, 0.9, 1.4);

	seg.getEverythingOnTopOfTable(shr.outputCloud);

	seg.segmentFlat(seg.outputCloud);

	pcl::toROSMsg(*seg.outputCloud, *outPutCloud);

	pub.publish(outPutCloud);

	PointT center;
	cen.getCenter(seg.outputCloud, center);

	transform.setOrigin(tf::Vector3(center.x, center.y, center.z));
	transform.setRotation(tf::Quaternion(0.0, 0.0, 0.0));
	br.sendTransform(
			tf::StampedTransform(transform, startTime,
					"/head_mount_kinect_rgb_optical_frame", "pancake_oven"));

	ROS_INFO("done");

}
}

int main(int argc, char** argv) {
	// Initialize ROS
	ros::init(argc, argv, "oven_recognizer");
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

