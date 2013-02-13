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

//	typename pcl::PointCloud<PointT>::Ptr outputCloud;

	pcl::PointIndices::Ptr flatinliers;

	Segmentation(): flatinliers(new pcl::PointIndices)
	{

	}
	virtual ~Segmentation()
	{

	}

	bool extractFlat(typename pcl::PointCloud<PointT>::Ptr& inputCloud,
			typename pcl::PointCloud<PointT>::Ptr& outputCloud, bool setNeg, bool setKeepOrganized)
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
		seg.segment(*flatinliers, *coefficients);

		extract.setKeepOrganized(true);
		extract.setInputCloud(inputCloud);
		extract.setIndices(flatinliers);

		// invert filter

		typename pcl::PointCloud<PointT>::Ptr plane_cloud(new pcl::PointCloud<PointT>);
		extract.setNegative(false);
		extract.filter(*plane_cloud);

		typename pcl::search::KdTree<PointT>::Ptr tree(
				new pcl::search::KdTree<PointT>);

		std::vector<pcl::PointIndices> cluster_indices;
		pcl::EuclideanClusterExtraction<PointT> ec;
		ec.setClusterTolerance(0.02); // 2cm
		ec.setMinClusterSize(500);
		ec.setSearchMethod(tree);
		ec.setInputCloud(plane_cloud);
		ec.extract(cluster_indices);

		if (cluster_indices.size() == 1)
		{
			pcl::PointIndices::Ptr segment_inliers(new pcl::PointIndices);

			segment_inliers->indices.insert(segment_inliers->indices.end(),
					cluster_indices.begin()->indices.begin(), cluster_indices.begin()->indices.end());

			extract.setKeepOrganized(setKeepOrganized);
			extract.setInputCloud(inputCloud);
			extract.setIndices(segment_inliers);

			// invert filter
			extract.setNegative(setNeg);
			extract.filter(*outputCloud);
		}

		return cluster_indices.size() == 1;
	}

	bool getEverythingOnTopOfTable(
			typename pcl::PointCloud<PointT>::Ptr& inputCloud, float minHeight, float maxHeight)
	{

		typename pcl::PointCloud<PointT>::Ptr cloud_filtered(
				new pcl::PointCloud<PointT>);

		// Coefficients and inliers for plane segmentation
		pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr hullInliers(new pcl::PointIndices);

		/*
		 *  Filter out biggest plane
		 */

		typename pcl::PointCloud<PointT>::Ptr planeCloud(
				new pcl::PointCloud<PointT>);

		extractFlat(inputCloud, planeCloud, false, false);
		/*
		 * Get convex hull
		 */

		if (planeCloud->size() > 10000)
		{
			typename pcl::PointCloud<PointT>::Ptr hull_cloud(
					new pcl::PointCloud<PointT>);
			pcl::ConvexHull<PointT> chull;
			chull.setInputCloud(planeCloud);
			chull.reconstruct(*hull_cloud);

			/*
			 * Get everything on top of table
			 */

			pcl::ExtractPolygonalPrismData<PointT> ex;
			ex.setInputCloud(inputCloud);
			ex.setHeightLimits(minHeight, maxHeight);
			ex.setInputPlanarHull(hull_cloud);
			ex.segment(*hullInliers);

			pcl::ExtractIndices<PointT> objectsOnTableFilter;
			objectsOnTableFilter.setInputCloud(inputCloud);
			objectsOnTableFilter.setIndices(hullInliers);
			objectsOnTableFilter.filter(*inputCloud);
		}

		return inputCloud->size() > 100 && planeCloud->size() > 10000;
	}

	bool extractBigObjects(typename pcl::PointCloud<PointT>::Ptr& inputCloud)
	{
		// Creating the KdTree object for the search method of the extraction
		typename pcl::search::KdTree<PointT>::Ptr tree(
				new pcl::search::KdTree<PointT>);

		typename pcl::PointCloud<PointT>::Ptr cloudWithoutPlane(
				new pcl::PointCloud<PointT>);

		if (inputCloud->size() > 100)
		{
			if (!extractFlat(inputCloud, cloudWithoutPlane, true, true))
			{
				return false;
			}

			//ROS_ERROR("tree: %d", cloudWithoutPlane->size());
			tree->setInputCloud(cloudWithoutPlane);

			//ROS_ERROR("tree end");
			std::vector<pcl::PointIndices> cluster_indices;
			pcl::EuclideanClusterExtraction<PointT> ec;
			ec.setClusterTolerance(0.02); // 2cm
			ec.setMinClusterSize(100);
			ec.setMaxClusterSize(25000);
			ec.setSearchMethod(tree);
			ec.setInputCloud(cloudWithoutPlane);
			ec.extract(cluster_indices);

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
				extract.setKeepOrganized(true);
				extract.setInputCloud(inputCloud);
				extract.setIndices(box_inliers);

				// invert filter
				extract.setNegative(true);
				extract.filter(*inputCloud);

			}

		}
		else
		{
			return false;
		}

		return true;

	}

	bool extractColors(typename pcl::PointCloud<PointT>::Ptr& inputCloud, std::vector<typename pcl::PointCloud<PointT>::Ptr>& outputVector)
	{
		typename pcl::search::Search<PointT>::Ptr tree = boost::shared_ptr<pcl::search::Search<PointT> >(
				new pcl::search::KdTree<PointT>);

		pcl::RegionGrowingRGB<PointT> reg;
		reg.setInputCloud(inputCloud);
		reg.setSearchMethod(tree);
		reg.setDistanceThreshold(1);
		reg.setPointColorThreshold(6);
		reg.setRegionColorThreshold(5);
		reg.setMinClusterSize(50);
		reg.setMaxClusterSize(500);

		std::vector<pcl::PointIndices> clusters;
		reg.extract(clusters);


		ROS_ERROR("Colorcluster found: %d", clusters.size());
		for (std::vector<pcl::PointIndices>::const_iterator it =
				clusters.begin(); it != clusters.end(); ++it)
		{
			pcl::PointIndices::Ptr color_inliers(new pcl::PointIndices);
			typename pcl::PointCloud<PointT>::Ptr segment_cloud(new pcl::PointCloud<PointT>);

			color_inliers->indices.insert(color_inliers->indices.end(),
					clusters.begin()->indices.begin(), clusters.begin()->indices.end());
			pcl::ExtractIndices<PointT> coloredObjects;
			coloredObjects.setInputCloud(inputCloud);
			coloredObjects.setIndices(color_inliers);
			coloredObjects.filter(*segment_cloud);

			outputVector.push_back(segment_cloud);
			ROS_ERROR("size: %d", segment_cloud->size());
		}

		return true;
	}
};

#endif /* SEGMENTATION_H_ */
