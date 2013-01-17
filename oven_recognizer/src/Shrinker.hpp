/*
 * Shrinker.h
 *
 *  Created on: 17.01.2013
 *      Author: nico
 */

#ifndef SHRINKER_H_
#define SHRINKER_H_

#include <pcl/filters/conditional_removal.h>

class Shrinker {
public:
	Shrinker() {

	}
	virtual ~Shrinker() {

	}

	bool resizeTo(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& inputCloud,
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr& outputCloud, float zMin,
			float zMax) {
		pcl::ConditionAnd<pcl::PointXYZRGB>::Ptr range_cond(
				new pcl::ConditionAnd<pcl::PointXYZRGB>());

		range_cond->addComparison(
				pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(
						new pcl::FieldComparison<pcl::PointXYZRGB>("z",
								pcl::ComparisonOps::GT, zMin)));
		range_cond->addComparison(
				pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(
						new pcl::FieldComparison<pcl::PointXYZRGB>("z",
								pcl::ComparisonOps::LT, zMax)));

		// build the filter
		pcl::ConditionalRemoval<pcl::PointXYZRGB> condrem(range_cond);
		condrem.setInputCloud(inputCloud);
		// apply filter
		condrem.filter(*outputCloud);

		return false;
	}
};

#endif /* SHRINKER_H_ */
