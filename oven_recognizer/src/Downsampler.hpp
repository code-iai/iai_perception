/*
 * Downsampler.h
 *
 *  Created on: 18.01.2013
 *      Author: nico
 */

#ifndef DOWNSAMPLER_H_
#define DOWNSAMPLER_H_
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
template<typename PointT> class Downsampler {
public:
	Downsampler():outputCloud(new pcl::PointCloud<PointT>){

	}
	virtual ~Downsampler(){

	}

	bool downsample(typename pcl::PointCloud<PointT>::Ptr inputCloud) {

		// Create the filtering object
		typename pcl::VoxelGrid<PointT> sor;
		sor.setInputCloud(inputCloud);
		sor.setLeafSize(0.006f, 0.006f, 0.006f);
		sor.filter(*outputCloud);

		return true;
	}

	typename pcl::PointCloud<PointT>::Ptr outputCloud;
};

#endif /* DOWNSAMPLER_H_ */
