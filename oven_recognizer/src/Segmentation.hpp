/*
 * Segmentation.h
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

typedef pcl::PointXYZRGB PointT;

class Segmentation {
public:

	Segmentation() {

	}
	virtual ~Segmentation() {

	}

	bool segmentFlat(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& inputCloud,
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr& outputCloud) {

		pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
		pcl::ExtractIndices<pcl::PointXYZRGB> extract;
		// Create the segmentation object
		pcl::SACSegmentation<pcl::PointXYZRGB> seg;
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

	bool segmentCircle(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& inputCloud,
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr& outputCloud) {

		  pcl::NormalEstimation<PointT, pcl::Normal> ne;
		  pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
		  pcl::ExtractIndices<PointT> extract;
		  pcl::ExtractIndices<pcl::Normal> extract_normals;
		  pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());

		  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
		  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2 (new pcl::PointCloud<pcl::Normal>);
		  pcl::ModelCoefficients::Ptr coefficients_plane (new pcl::ModelCoefficients), coefficients_cylinder (new pcl::ModelCoefficients);
		  pcl::PointIndices::Ptr inliers_plane (new pcl::PointIndices), inliers_cylinder (new pcl::PointIndices);

		  // Estimate point normals
		  ne.setSearchMethod (tree);
		  ne.setInputCloud (inputCloud);
		  ne.setKSearch (50);
		  ne.compute (*cloud_normals);


		  // Create the segmentation object for circle segmentation and set all the parameters
		  seg.setOptimizeCoefficients (true);
		  seg.setModelType (pcl::SACMODEL_CIRCLE2D);
		  seg.setMethodType (pcl::SAC_RANSAC);
		  seg.setNormalDistanceWeight (0.1);
		  seg.setMaxIterations (1000);
		  seg.setDistanceThreshold (0.1);
		  seg.setRadiusLimits (0.08, 0.14);
		  seg.setInputCloud (inputCloud);
		  seg.setInputNormals (cloud_normals);

		  // Obtain the circle inliers and coefficients
		  seg.segment (*inliers_cylinder, *coefficients_cylinder);

		  // Write the cylinder inliers to disk
		  extract.setInputCloud (inputCloud);
		  extract.setIndices (inliers_cylinder);
		  extract.setNegative (false);

		  extract.filter (*outputCloud);
		  if (outputCloud->points.empty ()){
		    std::cerr << "Can't find the cylindrical component." << std::endl;
		    return false;
		  }

		  return true;
	}

};

#endif /* SEGMENTATION_H_ */
