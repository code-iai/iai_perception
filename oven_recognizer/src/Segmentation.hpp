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

template<typename PointT> class Segmentation {
public:

	typename pcl::PointCloud<PointT>::Ptr outputCloud;

	Segmentation() :
			outputCloud(new pcl::PointCloud<PointT>) {

	}
	virtual ~Segmentation() {

	}

	bool segmentFlat(typename pcl::PointCloud<PointT>::Ptr& inputCloud,
			pcl::ModelCoefficients::Ptr& coefficients) {

		pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
		pcl::ExtractIndices<PointT> extract;
		// Create the segmentation object
		pcl::SACSegmentation<PointT> seg;
		// Optional
		seg.setOptimizeCoefficients(true);
		// Mandatory
		seg.setModelType(pcl::SACMODEL_PLANE);
		seg.setMethodType(pcl::SAC_RANSAC);
		seg.setDistanceThreshold(0.005);

		seg.setInputCloud(inputCloud);
		seg.segment(*inliers, *coefficients);

	//	extract.setKeepOrganized(true);
		extract.setInputCloud(inputCloud);
		extract.setIndices(inliers);

		// invert filter
		extract.setNegative(false);
		extract.filter(*outputCloud);

		return true;
	}

	bool segmentCircle(typename pcl::PointCloud<PointT>::Ptr& inputCloud) {

		pcl::NormalEstimation<PointT, pcl::Normal> ne;
		pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
		pcl::ExtractIndices<PointT> extract;
		pcl::ExtractIndices<pcl::Normal> extract_normals;
		typename pcl::search::KdTree<PointT>::Ptr tree(
				new pcl::search::KdTree<PointT>());

		pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(
				new pcl::PointCloud<pcl::Normal>);
		pcl::ModelCoefficients::Ptr coefficients_cylinder(
				new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices),
				inliers_cylinder(new pcl::PointIndices);

		// Estimate point normals
		ne.setSearchMethod(tree);
		ne.setInputCloud(inputCloud);
		ne.setKSearch(50);
		ne.compute(*cloud_normals);

		// Create the segmentation object for circle segmentation and set all the parameters
		seg.setOptimizeCoefficients(true);
		seg.setModelType(pcl::SACMODEL_CIRCLE2D);
		seg.setMethodType(pcl::SAC_RANSAC);
		seg.setNormalDistanceWeight(0.1);
		seg.setMaxIterations(1000);
		seg.setDistanceThreshold(0.1);
		seg.setRadiusLimits(0.08, 0.12);
		seg.setInputCloud(inputCloud);
		seg.setInputNormals(cloud_normals);

		// Obtain the circle inliers and coefficients
		seg.segment(*inliers_cylinder, *coefficients_cylinder);

		// Write the cylinder inliers to disk
		extract.setInputCloud(inputCloud);
		extract.setIndices(inliers_cylinder);
		extract.setNegative(false);

		extract.filter(*outputCloud);
		if (outputCloud->points.empty()) {
			std::cerr << "Can't find the circle component." << std::endl;
			return false;
		}

		return true;
	}

	bool getEverythingOnTopOfTable(
			typename pcl::PointCloud<PointT>::Ptr& inputCloud) {

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
		seg.setDistanceThreshold(0.01);

		seg.setInputCloud(inputCloud);
		seg.segment(*planeInliers, *coefficients);

		// Initializing with true will allow us to extract the removed indices
		pcl::ExtractIndices<PointT> eifilter;

		eifilter.setInputCloud(inputCloud);
		eifilter.setIndices(planeInliers);

		typename pcl::PointCloud<PointT>::Ptr planeCloud(
				new pcl::PointCloud<PointT>);

		eifilter.filter(*planeCloud);

		/*
		 * Get convex hull
		 */

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
		ex.setHeightLimits(0.03, 0.2);
		ex.setInputPlanarHull(hull_cloud);
		ex.segment(*chullInliers);

		pcl::ExtractIndices<PointT> objectsOnTableFilter;
		objectsOnTableFilter.setInputCloud(inputCloud);
		objectsOnTableFilter.setIndices(chullInliers);
		objectsOnTableFilter.filter(*outputCloud);

		return true;
	}
};

#endif /* SEGMENTATION_H_ */
