#include <iostream>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
class Segmentation 
{
   private:
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_;
    
    public:
        Segmentation();
        ~Segmentation();
        
        virtual void addCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr) =0;
        virtual void segment ()=0;
};
