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
namespace ovenRecognizer
{
std::vector<ros::Publisher> pub;
Segmentation<PointT> seg;
Shrinker<PointT> shr;
Downsampler<PointT> dwn;

void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& cloud)
{
	ros::Time begin = ros::Time::now();
	pcl::PointCloud<PointT>::Ptr inputCloud(new pcl::PointCloud<PointT>);
	const sensor_msgs::PointCloud2::Ptr outPutCloud(
			new sensor_msgs::PointCloud2);

	pcl::fromROSMsg(*cloud, *inputCloud);

	ros::Time downsamplebegin = ros::Time::now();
	dwn.downsample(inputCloud, 0.006f);
	ROS_INFO("downsamplebegin: %d", (ros::Time::now() - downsamplebegin).nsec);

	ros::Time resizebegin = ros::Time::now();
	shr.resizeTo(dwn.outputCloud, 0, 1.2);
	ROS_INFO("resizebegin: %d", (ros::Time::now() - resizebegin).nsec);

	ros::Time tabletopbegin = ros::Time::now();
	seg.getEverythingOnTopOfTable(shr.outputCloud, 0.0, 0.3);
	ROS_INFO("tabletopbegin: %d", (ros::Time::now() - tabletopbegin).nsec);

	pcl::toROSMsg(*seg.outputCloud, *outPutCloud);
	pub[9].publish(outPutCloud);

	ros::Time bigobjectsbegin = ros::Time::now();
	seg.extractBigObjects(seg.outputCloud);
	ROS_INFO("bigobjectsbegin: %d", (ros::Time::now() - bigobjectsbegin).nsec);

	std::vector<pcl::PointCloud<PointT>::Ptr> clusters;

	ros::Time tabletopbegin2 = ros::Time::now();
	seg.getEverythingOnTopOfTable(seg.outputCloud, 0.0, 0.02);
	ROS_INFO("tabletopbegin2: %d", (ros::Time::now() - tabletopbegin2).nsec);

	ros::Time extractcolorsbegin = ros::Time::now();
	seg.extractColors(seg.outputCloud, clusters);
	ROS_INFO("extractcolorsbegin: %d", (ros::Time::now() - extractcolorsbegin).nsec);

	ros::Duration push_rate(1, 0);
	int i = 0;
	for (std::vector<pcl::PointCloud<PointT>::Ptr>::iterator it = clusters.begin(); it != clusters.end(); ++it)
	{
		pcl::toROSMsg(*(*it), *outPutCloud);

		pub[i].publish(outPutCloud);

		++i;
	}

	ROS_INFO("Duration: %d", (ros::Time::now() - begin).nsec);
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

