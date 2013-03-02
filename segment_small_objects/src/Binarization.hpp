/*
 * Binaryzation.hpp
 *
 *  Created on: Feb 13, 2013
 *      Author: jalapeno
 */

#ifndef BINARIZATION_HPP_
#define BINARIZATION_HPP_

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

typedef pcl::PointXYZRGB PointT;

class Binarization
{
public:
	Binarization()
	{

	}
	virtual ~Binarization()
	{

	}

	bool binarize(typename pcl::PointCloud<PointT>::Ptr inputCloud, float scale, float threshold)
	{

		if(inputCloud->size() <= 0){
			return false;
		}
		float gray;
		pcl::PointIndices::Ptr segment_inliers(new pcl::PointIndices);

		int i = 0;


		for (std::vector<PointT, Eigen::aligned_allocator_indirection<PointT> >::iterator
		it = inputCloud->points.begin();
				it != inputCloud->points.end(); it++, i++)
		{
			gray = (scaleColor(it->r,scale) + scaleColor(it->g,scale)  +scaleColor(it->b,scale) ) / 3;

			if (gray > threshold)
			{
				segment_inliers->indices.push_back(i);

			}

		}

		pcl::ExtractIndices<PointT> extract;
		extract.setKeepOrganized(true);
		extract.setInputCloud(inputCloud);
		extract.setNegative(true);
		extract.setIndices(segment_inliers);
		extract.filter(*inputCloud);

		return true;
	}

	int scaleColor(int color, float scale){
		return color * scale > 255 ? 255 : color * scale;
	}
};

#endif /* BINARYZATION_HPP_ */
