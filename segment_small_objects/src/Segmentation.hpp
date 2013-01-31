/*
 * Segmentation.hpp
 *
 *  Created on: 14.01.2013
 *      Author: nico
 */

#ifndef SEGMENTATION_H_
#define SEGMENTATION_H_

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>
#include <pcl/segmentation/region_growing_rgb.h>

template<typename PointT> class Segmentation
{
public:

	typename pcl::PointCloud<PointT>::Ptr outputCloud;

	Segmentation() :
			outputCloud(new pcl::PointCloud<PointT>)
	{

	}
	virtual ~Segmentation()
	{

	}

	bool extractFlat(typename pcl::PointCloud<PointT>::Ptr& inputCloud,
			typename pcl::PointCloud<PointT>::Ptr& outputCloud,
			pcl::PointIndices::Ptr& inliers)
	{

		pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

		pcl::ExtractIndices<PointT> extract;
		// Create the segmentation object
		pcl::SACSegmentation<PointT> seg;
		// Optional
		seg.setOptimizeCoefficients(true);
		// Mandatory
		seg.setModelType(pcl::SACMODEL_PLANE);
		seg.setMethodType(pcl::SAC_RANSAC);
		seg.setDistanceThreshold(0.01);

		seg.setInputCloud(inputCloud);
		seg.segment(*inliers, *coefficients);

		extract.setKeepOrganized(true);
		extract.setInputCloud(inputCloud);
		extract.setIndices(inliers);

		// invert filter
		extract.setNegative(true);
		extract.filter(*outputCloud);

		return true;
	}

	bool getEverythingOnTopOfTable(
			typename pcl::PointCloud<PointT>::Ptr& inputCloud)
	{

		typename pcl::PointCloud<PointT>::Ptr cloud_filtered(
				new pcl::PointCloud<PointT>);

		// Coefficients and inliers for plane segmentation
		pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr planeInliers(new pcl::PointIndices);
		pcl::PointIndices::Ptr chullInliers(new pcl::PointIndices);

		/*
		 *  Filter out biggest plane
		 */

		// Create the segmentation object
		pcl::SACSegmentation<PointT> seg;
		// Optional
		seg.setOptimizeCoefficients(true);
		// Mandatory
		seg.setModelType(pcl::SACMODEL_PLANE);
		seg.setMethodType(pcl::SAC_RANSAC);
		seg.setDistanceThreshold(0.03);

		seg.setInputCloud(inputCloud);
		seg.segment(*planeInliers, *coefficients);

		// Initializing with true will allow us to extract the removed indices
		pcl::ExtractIndices<PointT> eifilter(true);

		eifilter.setInputCloud(inputCloud);
		eifilter.setIndices(planeInliers);

		typename pcl::PointCloud<PointT>::Ptr planeCloud(
				new pcl::PointCloud<PointT>);

		eifilter.filter(*planeCloud);

		/*
		 * Get convex hull
		 */

		ROS_ERROR("table hull begin");
		typename pcl::PointCloud<PointT>::Ptr hull_cloud(
				new pcl::PointCloud<PointT>);
		pcl::ConvexHull<PointT> chull;
		chull.setInputCloud(planeCloud);
		chull.reconstruct(*hull_cloud);

		ROS_ERROR("table hull end");
		/*
		 * Get everything on top of table
		 */

		ROS_ERROR("extract begin");
		pcl::ExtractPolygonalPrismData<PointT> ex;
		ex.setInputCloud(inputCloud);
		ex.setHeightLimits(-0.3, 0.5);
		ex.setInputPlanarHull(hull_cloud);
		ex.segment(*chullInliers);

		ROS_ERROR("extract end");
		pcl::ExtractIndices<PointT> objectsOnTableFilter;
		objectsOnTableFilter.setInputCloud(inputCloud);
		objectsOnTableFilter.setIndices(chullInliers);
		objectsOnTableFilter.filter(*outputCloud);

		ROS_ERROR("bla");
		return true;
	}

	bool extractBigObjects(typename pcl::PointCloud<PointT>::Ptr& inputCloud)
	{
		ROS_ERROR("begin bigobj");
		// Creating the KdTree object for the search method of the extraction
		typename pcl::search::KdTree<PointT>::Ptr tree(
				new pcl::search::KdTree<PointT>);

		typename pcl::PointCloud<PointT>::Ptr cloudWithoutPlane(
				new pcl::PointCloud<PointT>);

		pcl::PointIndices::Ptr planeIndices(new pcl::PointIndices);

		ROS_ERROR("begin extractflat");
		extractFlat(inputCloud, cloudWithoutPlane, planeIndices);

		tree->setInputCloud(cloudWithoutPlane);

		ROS_ERROR("begin extract");
		std::vector<pcl::PointIndices> cluster_indices;
		pcl::EuclideanClusterExtraction<PointT> ec;
		ec.setClusterTolerance(0.02); // 2cm
		ec.setMinClusterSize(100);
		ec.setMaxClusterSize(25000);
		ec.setSearchMethod(tree);
		ec.setInputCloud(cloudWithoutPlane);
		ec.extract(cluster_indices);

		ROS_ERROR("begin iterate");
		for (std::vector<pcl::PointIndices>::const_iterator it =
				cluster_indices.begin(); it != cluster_indices.end(); ++it)
		{

			pcl::PointIndices::Ptr segment_inliers(new pcl::PointIndices);
			typename pcl::PointCloud<PointT>::Ptr segment_cloud(new pcl::PointCloud<PointT>);

			segment_inliers->indices.insert(segment_inliers->indices.end(),
					it->indices.begin(), it->indices.end());

			pcl::ExtractIndices<PointT> extract;
			extract.setKeepOrganized(true);
			extract.setInputCloud(inputCloud);
			extract.setIndices(segment_inliers);

			// invert filter
			extract.setNegative(false);
			extract.filter(*segment_cloud);

			Eigen::Vector4f minPoint;
			Eigen::Vector4f maxPoint;
			pcl::getMinMax3D(*segment_cloud, minPoint, maxPoint);

			for (int i = 0; i < 3; i++)
			{
				minPoint(i) -= 0.02;
				maxPoint(i) += 0.02;
			}

			pcl::PointIndices::Ptr box_inliers(new pcl::PointIndices);

			pcl::getPointsInBox(*inputCloud, minPoint, maxPoint, box_inliers->indices);

			//pcl::ExtractIndices<PointT> extract;
			extract.setKeepOrganized(true);
			extract.setInputCloud(inputCloud);
			extract.setIndices(box_inliers);

			// invert filter
			extract.setNegative(true);
			extract.filter(*inputCloud);
			//outputCloud = segment_cloud;

		}

		outputCloud = inputCloud;

		return true;

	}

	bool extractColors(typename pcl::PointCloud<PointT>::Ptr& inputCloud)
	{
		ROS_ERROR("begin colors");
		typename pcl::search::Search<PointT>::Ptr tree = boost::shared_ptr<pcl::search::Search<PointT> >(
				new pcl::search::KdTree<PointT>);

		pcl::RegionGrowingRGB<PointT> reg;
		reg.setInputCloud(inputCloud);
		reg.setSearchMethod(tree);
		reg.setDistanceThreshold(10);
		reg.setPointColorThreshold(6);
		reg.setRegionColorThreshold(5);
		reg.setMinClusterSize(100);
		reg.setMaxClusterSize(500);

		std::vector<pcl::PointIndices> clusters;
		reg.extract(clusters);

		ROS_ERROR("colorclustersize : %d", clusters.size());

		if(!clusters.size()){
			return false;
		}

		pcl::PointIndices::Ptr color_inliers(new pcl::PointIndices);

		color_inliers->indices.insert(color_inliers->indices.end(),
				clusters.begin()->indices.begin(), clusters.begin()->indices.end());
		pcl::ExtractIndices<PointT> coloredObjects;
		coloredObjects.setInputCloud(inputCloud);
		coloredObjects.setIndices(color_inliers);
		coloredObjects.filter(*outputCloud);

		return true;
	}
};

#endif /* SEGMENTATION_H_ */
