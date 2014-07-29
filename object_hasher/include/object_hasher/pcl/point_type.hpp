/*
 * point_type.hpp
 *
 *  Created on: Jul 29, 2011
 *      Author: ferka
 */

#ifndef POINT_TYPE_HPP_
#define POINT_TYPE_HPP_

namespace pcl {


struct PointNormalRADII
{
  PCL_ADD_POINT4D;    // This adds the members x,y,z which can also be accessed using the point (which is float[4])
  PCL_ADD_NORMAL4D;   // This adds the member normal[3] which can also be accessed using the point (which is float[4])
  float curvature, r_min, r_max;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
inline std::ostream& operator << (std::ostream& os, const PointNormalRADII& p)
{
  os << "(" << p.x << "," << p.y << "," << p.z << " - " << p.normal[0] << "," << p.normal[1] << "," << p.normal[2] << " - " << p.curvature << " - " << p.r_min << "," << p.r_max << ")";
  return (os);
}

struct PointXYZRGBNormalRegion
{
  PCL_ADD_POINT4D;    // This adds the members x,y,z which can also be accessed using the point (which is float[4])
  PCL_ADD_NORMAL4D;   // This adds the member normal[3] which can also be accessed using the point (which is float[4])
  union
  {
    struct
    {
      float rgb;
      float curvature;
      float reg;
    };
    float data_c[5];
  };
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
inline std::ostream& operator << (std::ostream& os, const PointXYZRGBNormalRegion& p)
{
  os << "(" << p.x << "," << p.y << "," << p.z << " - " << p.normal[0] << "," << p.normal[1] << "," << p.normal[2] <<","<<p.rgb<< " - " << p.curvature << " - "<<p.reg<< ")";
  return (os);
}

struct PointXYZConfidenceRegion
{
  PCL_ADD_POINT4D;
  float reg;
  float sphere, flat, box, cylinder,other,plate;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}EIGEN_ALIGN16;
inline std::ostream& operator << (std::ostream& os, const PointXYZConfidenceRegion& p)
{
  os << "(" << p.x << "," << p.y << "," << p.z << " - " << p.reg << " - " << ")";
  return (os);
}

struct PointXYZLRegion
{
  PCL_ADD_POINT4D;
  float reg;
  float label;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}EIGEN_ALIGN16;
inline std::ostream& operator << (std::ostream& os, const PointXYZLRegion& p)
{
  os << "(" << p.x << "," << p.y << "," << p.z << " - " << p.reg << " - " << p.label <<")";
  return (os);
}

struct EIGEN_ALIGN16 _PointXYZRGBNormalI
{
  PCL_ADD_POINT4D; // This adds the members x,y,z which can also be accessed using the point (which is float[4])
  PCL_ADD_NORMAL4D; // This adds the member normal[3] which can also be accessed using the point (which is float[4])
  union
  {
    struct
    {
      // RGB union
      union
      {
        struct
        {
          uint8_t b;
          uint8_t g;
          uint8_t r;
          uint8_t _unused;
        };
        float rgb;
        uint32_t rgba;
      };
      float intensity;
      float curvature;
    };
    float data_c[4];
  };
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct PointXYZRGBNormalI : public _PointXYZRGBNormalI
{
  inline PointXYZRGBNormalI ()
  {
    _unused = 0;
    data[3] = 1.0f;
  }

  inline Eigen::Vector3i getRGBVector3i () { return (Eigen::Vector3i (r, g, b)); }
  inline const Eigen::Vector3i getRGBVector3i () const { return (Eigen::Vector3i (r, g, b)); }
  inline Eigen::Vector4i getRGBVector4i () { return (Eigen::Vector4i (r, g, b, 0)); }
  inline const Eigen::Vector4i getRGBVector4i () const { return (Eigen::Vector4i (r, g, b, 0)); }
};

inline std::ostream& operator << (std::ostream& os, const PointXYZRGBNormalI& p)
{
  os << "(" << p.x << "," << p.y << "," << p.z << " - " << p.rgb << " - " << p.r
     << ", " << p.g << ", " << p.b << " - " << p.normal[0] << ","
     << p.normal[1] << "," << p.normal[2] << " - " << p.curvature << " - "
     << p.intensity << ")";
  return (os);
}


}
#endif /* POINT_TYPE_HPP_ */
