/*
 * point_type.h
 *
 *  Created on: Jul 29, 2011
 *      Author: ferka
 */

#ifndef POINT_TYPE_H_
#define POINT_TYPE_H_

#include <Eigen/Core>
#include <bitset>
#include <vector>
#include "pcl/register_point_struct.h"
#include <pcl/point_types.h>

#include "object_hasher/point_type.hpp"

namespace pcl {
	struct PointNormalRADII;
	struct PointXYZRGBNormalRegion;
	struct PointXYZConfidenceRegion;
	struct PointXYZRGBNormalI;
	struct PointXYZLRegion;
}

POINT_CLOUD_REGISTER_POINT_STRUCT (pcl::PointNormalRADII,
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, normal_x, normal_x)
                                   (float, normal_y, normal_y)
                                   (float, normal_z, normal_z)
                                   (float, curvature, curvature)
                                   (float, r_min, r_min)
                                   (float, r_max, r_max)
);
POINT_CLOUD_REGISTER_POINT_STRUCT (pcl::PointXYZRGBNormalRegion,
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, normal_x, normal_x)
                                   (float, normal_y, normal_y)
                                   (float, normal_z, normal_z)
                                   (float, rgb, rgb)
                                   (float, curvature, curvature)
                                   (float, reg, reg)
);

POINT_CLOUD_REGISTER_POINT_STRUCT (pcl::PointXYZConfidenceRegion,
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, reg, reg)
                                   (float, sphere, sphere)
                                   (float, box, box)
                                   (float, flat,flat)
                                   (float, cylinder, cylinder)
                                   (float, plate, plate)
                                   (float, other, other)
);
POINT_CLOUD_REGISTER_POINT_STRUCT (pcl::PointXYZLRegion,
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, reg, reg)
                                   (float, label, label)
);

POINT_CLOUD_REGISTER_POINT_STRUCT (pcl::_PointXYZRGBNormalI,
   (float, x, x)
   (float, y, y)
   (float, z, z)
   (float, rgb, rgb)
   (float, normal_x, normal_x)
   (float, normal_y, normal_y)
   (float, normal_z, normal_z)
   (float, curvature, curvature)
   (float, intensity, intensity)
)POINT_CLOUD_REGISTER_POINT_WRAPPER(pcl::PointXYZRGBNormalI,
                                    pcl::_PointXYZRGBNormalI)


#endif /* POINT_TYPE_H_ */
