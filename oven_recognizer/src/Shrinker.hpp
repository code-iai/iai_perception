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
		typename pcl::ConditionAnd<PointT>::Ptr range_cond(
				new pcl::ConditionAnd<PointT>());

		range_cond->addComparison(
				typename pcl::FieldComparison<PointT>::ConstPtr(
						new pcl::FieldComparison<PointT>("z",
								pcl::ComparisonOps::GT, zMin)));
		range_cond->addComparison(
				typename pcl::FieldComparison<PointT>::ConstPtr(
						new pcl::FieldComparison<PointT>("z",
								pcl::ComparisonOps::LT, zMax)));

		range_cond->addComparison(
				typename pcl::FieldComparison<PointT>::ConstPtr(
						new pcl::FieldComparison<PointT>("x",
								pcl::ComparisonOps::GT, -2.0f)));
		range_cond->addComparison(
				typename pcl::FieldComparison<PointT>::ConstPtr(
						new pcl::FieldComparison<PointT>("x",
								pcl::ComparisonOps::LT, 2.0f)));


		range_cond->addComparison(
				typename pcl::FieldComparison<PointT>::ConstPtr(
						new pcl::FieldComparison<PointT>("y",
								pcl::ComparisonOps::GT, -2.0f)));
		range_cond->addComparison(
				typename pcl::FieldComparison<PointT>::ConstPtr(
						new pcl::FieldComparison<PointT>("y",
								pcl::ComparisonOps::LT, 2.0f)));

		// build the filter
		pcl::ConditionalRemoval<PointT> condrem(range_cond);
		condrem.setInputCloud(inputCloud);
		// apply filter
		condrem.filter(*outputCloud);

		return false;
	}
};

#endif /* SHRINKER_H_ */
