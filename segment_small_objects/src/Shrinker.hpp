/*
 * Shrinker.h
 *
 *  Created on: 17.01.2013
 *      Author: nico
 */

#ifndef SHRINKER_H_
#define SHRINKER_H_

#include <pcl/filters/conditional_removal.h>

template<typename PointT> class Shrinker {
public:
	typename pcl::PointCloud<PointT>::Ptr outputCloud;
	Shrinker():outputCloud(new pcl::PointCloud<PointT>) {

	}
	virtual ~Shrinker() {

	}

	bool resizeTo(typename pcl::PointCloud<PointT>::Ptr& inputCloud, float zMin,
			float zMax) {
		pcl::PassThrough<PointT> pass;
		pass.setInputCloud(inputCloud);
		pass.setFilterFieldName("z");
		pass.setFilterLimits(0.0, 1.0);
		pass.filter(*outputCloud);

		return true;
	}
};

#endif /* SHRINKER_H_ */
