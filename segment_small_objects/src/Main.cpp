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

bool segment_small_objects(const sensor_msgs::PointCloud2ConstPtr& cloud, std::vector<typename pcl::PointCloud<PointT>::Ptr>& clusters)
{
	DEBUG_TIME("Duration",
			pcl::PointCloud<PointT>::Ptr inputCloud(new pcl::PointCloud<PointT>);
			const sensor_msgs::PointCloud2::Ptr outPutCloud(
			new sensor_msgs::PointCloud2);

			pcl::fromROSMsg(*cloud, *inputCloud);

			DEBUG_TIME("downsample", if(!dwn.downsample(inputCloud, 0.006f))return false;)

			inputCloud = dwn.outputCloud;

			DEBUG_TIME("Resize", if(!shr.resizeTo(inputCloud, 0, 2))return false;)

			DEBUG_TIME("tableTop", if(!seg.getEverythingOnTopOfTable(inputCloud, -0.01, 0.3))return false;)

			DEBUG_TIME("extractBigObjects", if(!seg.extractBigObjects(inputCloud, clusters))return false;)

			DEBUG_TIME("binarize", if(!bin.binarize(inputCloud, 1.5, 220))return false;)

			DEBUG_TIME("extractAllCluster",seg.extractAllObjects(inputCloud, clusters);)

			DEBUG_TIME("extractHullCollisions", seg.extractHullCollisions(clusters);)

			)
	return true;
}
}

