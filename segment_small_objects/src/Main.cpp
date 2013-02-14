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

//#define DEBUG
#include "TimeMeasureMacro.h"

#include <math.h>

typedef pcl::PointXYZRGB PointT;
namespace segmentSmallObjects
{
std::vector<ros::Publisher> pub;
Segmentation<PointT> seg;
Shrinker<PointT> shr;
Downsampler<PointT> dwn;
Binarization bin;

void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& cloud)
{
	DEBUG_TIME("Duration",
			pcl::PointCloud<PointT>::Ptr inputCloud(new pcl::PointCloud<PointT>);
			const sensor_msgs::PointCloud2::Ptr outPutCloud(
			new sensor_msgs::PointCloud2);

			pcl::fromROSMsg(*cloud, *inputCloud);

			DEBUG_TIME("downsample", if(!dwn.downsample(inputCloud, 0.006f))return;)

			inputCloud = dwn.outputCloud;

			DEBUG_TIME("Resize", if(!shr.resizeTo(inputCloud, 0, 1.2))return;)

			DEBUG_TIME("tableTop", if(!seg.getEverythingOnTopOfTable(inputCloud, -0.01, 0.3))return;)

//			pcl::toROSMsg(*seg.hull, *outPutCloud);
//						pub[7].publish(outPutCloud);

			DEBUG_TIME("extractBigObjects", if(!seg.extractBigObjects(inputCloud))return;)

//			pcl::toROSMsg(*inputCloud, *outPutCloud);
//						pub[8].publish(outPutCloud);

			DEBUG_TIME("binarize", if(!bin.binarize(inputCloud, 1.5, 220))return;)

//			pcl::toROSMsg(*inputCloud, *outPutCloud);
//			pub[9].publish(outPutCloud);

			std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;

			DEBUG_TIME("extractAllCluster",seg.extractAllObjects(inputCloud, clusters);)

			DEBUG_TIME("extractHullCollisions", seg.extractHullCollisions(clusters);)

			int i = 0;
			for (std::vector<pcl::PointCloud<PointT>::Ptr>::iterator it = clusters.begin(); it != clusters.end() && i != 9; ++it)
			{
			pcl::toROSMsg(*(*it), *outPutCloud);

			ROS_INFO("Cloud %d: %d",i, (*it)->size());
			pub[i].publish(outPutCloud);

			++i;
			}
			)
}
}

int main(int argc, char** argv)
{
	// Initialize ROS
	ros::init(argc, argv, "oven_recognizer");
	ros::NodeHandle nh;

	// Create a ROS subscriber for the input point cloud
	ros::Subscriber sub = nh.subscribe("/camera/depth_registered/points",
			1, segmentSmallObjects::cloud_cb);

	std::string publishername = "/piripiri/depth_registered/points";
	for (int i = 0; i < 10; i++)
	{
		segmentSmallObjects::pub.push_back(nh.advertise<sensor_msgs::PointCloud2>(
				publishername + boost::lexical_cast<std::string>(i), 1));
	}

	ros::Rate r(100);
	ros::spin();
	r.sleep();
	return 0;
}

