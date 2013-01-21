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

#include <math.h>

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
	//ros::Time startTime =
	tf::TransformBroadcaster br;
	tf::TransformListener listener;

	tf::StampedTransform transform;
	std::cerr<<"Input Cloud Size: " <<inputCloud->points.size()<<std::endl;
	dwn.downsample(inputCloud);

	shr.resizeTo(dwn.outputCloud, 0.9, 1.4);

	seg.getEverythingOnTopOfTable(shr.outputCloud);

	seg.segmentCircle(seg.outputCloud);

	pcl::ModelCoefficients::Ptr flatCoefficients(new pcl::ModelCoefficients);

	seg.segmentFlat(seg.outputCloud, flatCoefficients);

	pcl::toROSMsg(*seg.outputCloud, *outPutCloud);

	pub.publish(outPutCloud);

	PointT center;
	cen.getCenter(seg.outputCloud, center);
	std::cerr<<"Center of the oven: "<<center.x<<" "<<center.y<<" "<<center.z<<std::endl;
	float bla = std::sqrt((std::pow(flatCoefficients->values[0],2) +
			std::pow(flatCoefficients->values[1],2) +
			std::pow(flatCoefficients->values[2],2)));

	tf::StampedTransform transform_2;
	    try{
	      listener.lookupTransform("/base_link", "/head_mount_kinect_rgb_optical_frame",
	                               ros::Time(0), transform_2);
	    }
	    catch (tf::TransformException ex){
	      ROS_ERROR("%s",ex.what());
	    }
	tf::Quaternion orientation=transform_2.getRotation();



	transform.setOrigin(tf::Vector3(center.x, center.y, center.z));
	transform.setRotation(orientation);




	br.sendTransform(
			tf::StampedTransform(transform, ros::Time::now(),
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
	ros::Rate r(100);
	ros::spin();
	r.sleep();
	return 0;
}

