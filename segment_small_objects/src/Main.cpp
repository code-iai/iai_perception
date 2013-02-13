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
#include "Binarization.hpp"

#include <math.h>

typedef pcl::PointXYZRGB PointT;
namespace ovenRecognizer
{
std::vector<ros::Publisher> pub;
Segmentation<PointT> seg;
Shrinker<PointT> shr;
Downsampler<PointT> dwn;
Binarization bin;

void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& cloud)
{
	ros::Time begin = ros::Time::now();
	pcl::PointCloud<PointT>::Ptr inputCloud(new pcl::PointCloud<PointT>);
	const sensor_msgs::PointCloud2::Ptr outPutCloud(
			new sensor_msgs::PointCloud2);

	pcl::fromROSMsg(*cloud, *inputCloud);

	ros::Time downsamplebegin = ros::Time::now();
	dwn.downsample(inputCloud, 0.006f);
	ROS_INFO("downsamplebegin: %f", 0.000001 * (ros::Time::now() - downsamplebegin).nsec);

	pcl::toROSMsg(*dwn.outputCloud, *outPutCloud);
	pub[7].publish(outPutCloud);

	inputCloud = dwn.outputCloud;

	ros::Time resizebegin = ros::Time::now();
	shr.resizeTo(inputCloud, 0, 1.2);
	ROS_INFO("resizebegin: %f", 0.000001 * (ros::Time::now() - resizebegin).nsec);

	ros::Time tabletopbegin = ros::Time::now();
	seg.getEverythingOnTopOfTable(inputCloud, -0.01, 0.3);
	ROS_INFO("tabletopbegin: %f", 0.000001 * (ros::Time::now() - tabletopbegin).nsec);

	ros::Time bigobjectsbegin = ros::Time::now();
	seg.extractBigObjects(inputCloud);
	ROS_INFO("bigobjectsbegin: %f", 0.000001 * (ros::Time::now() - bigobjectsbegin).nsec);

	pcl::toROSMsg(*inputCloud, *outPutCloud);
	pub[8].publish(outPutCloud);
//	std::vector<pcl::PointCloud<PointT>::Ptr> clusters;
//
//	ros::Time tabletopbegin2 = ros::Time::now();
//	seg.getEverythingOnTopOfTable(seg.outputCloud, -0.02, 0.02);
//	ROS_INFO("tabletopbegin2: %f", 0.000001 * (ros::Time::now() - tabletopbegin2).nsec);

	ros::Time binarizebegin = ros::Time::now();
	bin.binarize(inputCloud, 1.5, 220);
	ROS_INFO("binarizebegin: %f", 0.000001 * (ros::Time::now() - binarizebegin).nsec);

	pcl::toROSMsg(*inputCloud, *outPutCloud);
	pub[9].publish(outPutCloud);

//	ros::Time extractcolorsbegin = ros::Time::now();
//	seg.extractColors(seg.outputCloud, clusters);
//	ROS_INFO("extractcolorsbegin: %d", (ros::Time::now() - extractcolorsbegin).nsec);
//
//
//	int i = 0;
//	for (std::vector<pcl::PointCloud<PointT>::Ptr>::iterator it = clusters.begin(); it != clusters.end(); ++it)
//	{
//		pcl::toROSMsg(*(*it), *outPutCloud);
//
//		pub[i].publish(outPutCloud);
//
//		++i;
//	}

	ROS_INFO("Duration: %f", 0.000001 * (ros::Time::now() - begin).nsec);
}
}

int main(int argc, char** argv)
{
	// Initialize ROS
	ros::init(argc, argv, "oven_recognizer");
	ros::NodeHandle nh;

	// Create a ROS subscriber for the input point cloud
	ros::Subscriber sub = nh.subscribe("/camera/depth_registered/points",
			1, ovenRecognizer::cloud_cb);

	std::string publishername = "/piripiri/depth_registered/points";
	for (int i = 0; i < 10; i++)
	{
		ovenRecognizer::pub.push_back(nh.advertise<sensor_msgs::PointCloud2>(
				publishername + boost::lexical_cast<std::string>(i), 1));
	}

	ros::Rate r(100);
	ros::spin();
	r.sleep();
	return 0;
}

