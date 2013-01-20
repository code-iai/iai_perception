/*
 * Downsampler.hpp
 *
 *  Created on: 18.01.2013
 *      Author: nico
 */

#ifndef CENTER_H_
#define CENTER_H_
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
template<typename PointT> class Center {
public:
	Center() {

	}
	virtual ~Center() {

	}

	bool getCenter(typename pcl::PointCloud<PointT>::Ptr& inputCloud,
			PointT& center) {
		PointT minX, maxX;
		minX = maxX = *(inputCloud->points.begin());

		for (typename pcl::PointCloud<PointT>::const_iterator it =
				inputCloud->points.begin(); it != inputCloud->points.end();
				++it) {
			if (minX.x > it->x) {
				minX = *it;
			} else if (maxX.x < it->x) {
				maxX = *it;
			}
		}

		center.x = (minX.x + maxX.x) / 2;
		center.y = (minX.y + maxX.y) / 2;
		center.z = (minX.z + maxX.z) / 2;

		return true;
	}

};

#endif /* CENTER_H_ */
