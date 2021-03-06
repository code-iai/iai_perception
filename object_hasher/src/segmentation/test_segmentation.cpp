
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>

#include "object_hasher/segmentation/euclidean_clustering_curvature.h"


typedef pcl::PointXYZRGBA PointT;

bool help(int argc, char *argv[])
{
  if(pcl::console::find_switch(argc, argv, "-h") || pcl::console::find_switch(argc, argv, "-?") || pcl::console::find_switch(argc, argv, "--help"))
  {
    std::string name = argv[0];
    name = name.substr(name.rfind('/') + 1);

    std::cout << name << " [options]" << std::endl
              << "Options:" << std::endl
              << "  -in    <file>        Input pcd filet" << std::endl
              << "  -out   <path>        Output Path"<<std::endl;
    return true;
  }
  else
  {
    return false;
  }
}


int main(int argc, char *argv[])
{
  if(help(argc, argv))
  {
    return 0;
  }
  std::string pathIn = "data/test.pcd";
  std::string pathOut = "./";
  pcl::console::parse(argc, argv, "-in", pathIn);
  pcl::console::parse(argc, argv, "-out", pathOut);

  pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
  pcl::PCDReader reader;
  pcl::PCDWriter writer;
  reader.read(pathIn,*cloud);
  std::cerr<<"Input Cloud has: "<<cloud->size()<<" points."<<std::endl;
  std::cerr<<"Region growing based on curvature started"<<std::endl;
  std::vector<pcl::PointIndices::Ptr> segments;
  oph::EuclideanClusterExtractorCurvature ecec;
  ecec.addCloud(cloud);
  ecec.setTrueRandom(true);
  ecec.segment(segments);

  std::cerr<<"Segments found: "<<segments.size()<<std::endl;

  pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_labeled(new pcl::PointCloud<pcl::PointXYZL>());
  pcl::copyPointCloud(*cloud,*cloud_labeled);
  for (unsigned int i = 0;i<segments.size(); ++i)
  {

    for(unsigned int j = 0;j<segments[i]->indices.size(); ++j)
    {
      cloud_labeled->points[(segments[i]->indices[j])].label = i;
    }
  }
  writer.write(pathOut+"cloud_labeled.pcd",*cloud_labeled);

}
