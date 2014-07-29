#include <iostream>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PointIndices.h>
class Segmentation 
{
protected:
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_;
public:

    /**
     * @brief Segmentation empty constructor
     */
    Segmentation()
    {

    }

    /**
     * @brief ~Segmentation empty destructor
     */
    virtual ~Segmentation()
    {

    }

    virtual void addCloud(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &) =0;
    virtual void segment (std::vector<pcl::PointIndices::Ptr>&)=0;
};
