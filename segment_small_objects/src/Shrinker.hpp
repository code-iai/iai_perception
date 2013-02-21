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
	Shrinker(){

	}
	virtual ~Shrinker() {

	}

	bool resizeTo(typename pcl::PointCloud<PointT>::Ptr& inputCloud, float zMin,
			float zMax) {
		pcl::PassThrough<PointT> pass;
		pass.setInputCloud(inputCloud);
		pass.setFilterFieldName("z");
		pass.setFilterLimits(zMin, zMax);
		pass.filter(*inputCloud);

		return true;
	}
};

#endif /* SHRINKER_H_ */
