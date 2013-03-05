#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/extract_clusters.h>

typedef pcl::PointXYZRGB PointT;

class Segmentator {
public:

	std::vector<pcl::PointCloud<PointT>::Ptr> clusters;

	bool segment(const pcl::PointCloud<PointT>::Ptr& planarCloud,
			const std::vector<pcl::PointIndices::Ptr>& objects,
			const pcl::PointCloud<PointT>::Ptr& planarHull) {

		clusters.clear();

		pcl::PointCloud<PointT>::Ptr inputCloud(
				new pcl::PointCloud<PointT>(*planarCloud));

		if (!binarize(inputCloud, 1.5, 200))
			return false;

		extractAllObjects(inputCloud, clusters);

		extractCollisions(clusters, planarHull);

		extractBigObjectCollisions(clusters, planarCloud, objects);

		return true;
	}

private:
	bool binarize(pcl::PointCloud<PointT>::Ptr inputCloud, const float scale,
			const float threshold) {

		if (inputCloud->size() <= 0) {
			return false;
		}
		float gray;
		pcl::PointIndices::Ptr segment_inliers(new pcl::PointIndices);

		int i = 0;

		for (std::vector<PointT, Eigen::aligned_allocator_indirection<PointT> >::iterator it =
				inputCloud->points.begin(); it != inputCloud->points.end();
				it++, i++) {
			gray = (scaleColor(it->r, scale) + scaleColor(it->g, scale)
					+ scaleColor(it->b, scale)) / 3;

			if (gray > threshold) {
				segment_inliers->indices.push_back(i);
			}

		}

		pcl::ExtractIndices<PointT> extract;
		extract.setKeepOrganized(true);
		extract.setInputCloud(inputCloud);
		extract.setNegative(true);
		extract.setIndices(segment_inliers);
		extract.filter(*inputCloud);

		return true;
	}

	inline int scaleColor(const int color, const float scale) const {
		return color * scale > 255 ? 255 : color * scale;
	}

	bool extractAllObjects(const pcl::PointCloud<PointT>::Ptr& inputCloud,
			std::vector<pcl::PointCloud<PointT>::Ptr>& outputVector) {

		pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);

		std::vector<pcl::PointIndices> cluster_indices;
		pcl::EuclideanClusterExtraction<PointT> ec;
		ec.setClusterTolerance(0.02); // 2cm
		ec.setMinClusterSize(40);
		ec.setMaxClusterSize(2000);
		ec.setSearchMethod(tree);
		ec.setInputCloud(inputCloud);
		ec.extract(cluster_indices);

		for (std::vector<pcl::PointIndices>::const_iterator it =
				cluster_indices.begin(); it != cluster_indices.end(); ++it) {
			pcl::PointIndices::Ptr color_inliers(new pcl::PointIndices);
			pcl::PointCloud<PointT>::Ptr segment_cloud(
					new pcl::PointCloud<PointT>);

			color_inliers->indices.insert(color_inliers->indices.end(),
					it->indices.begin(), it->indices.end());
			pcl::ExtractIndices<PointT> coloredObjects;
			coloredObjects.setInputCloud(inputCloud);
			coloredObjects.setIndices(color_inliers);
			coloredObjects.filter(*segment_cloud);

			outputVector.push_back(segment_cloud);
		}

		return true;
	}

	bool extractCollisions(
			std::vector<pcl::PointCloud<PointT>::Ptr>& cluster,
			const pcl::PointCloud<PointT>::Ptr& cloud) {

		pcl::KdTreeFLANN<PointT> kdtree;

		for (std::vector<pcl::PointCloud<PointT>::Ptr>::iterator it =
				cluster.begin(); it != cluster.end();) {
			bool erased = false;

			kdtree.setInputCloud(cloud);

			std::vector<PointT, Eigen::aligned_allocator_indirection<PointT> >::iterator it2 =
					(*it)->points.begin();
			std::vector<PointT, Eigen::aligned_allocator_indirection<PointT> >::iterator itend =
					(*it)->points.end();
			for (; it2 != itend; it2++) {

				// Neighbors within radius search

				std::vector<int> pointIdxRadiusSearch;
				std::vector<float> pointRadiusSquaredDistance;

				const float radius = 0.01f;

				if (kdtree.radiusSearch(*it2, radius, pointIdxRadiusSearch,
						pointRadiusSquaredDistance) > 0) {
					it = cluster.erase(it);
					erased = true;
					break;
				}
			}

			if (!erased) {
				it++;
			}
		}

		return true;
	}

	bool extractBigObjectCollisions(
			std::vector<pcl::PointCloud<PointT>::Ptr>& cluster,
			const pcl::PointCloud<PointT>::Ptr& planarCloud,
			const std::vector<pcl::PointIndices::Ptr>& bigObjects) {

		for (std::vector<pcl::PointIndices::Ptr>::const_iterator it =
				bigObjects.begin(); it != bigObjects.end(); ++it) {
			pcl::PointCloud<PointT>::Ptr segment_cloud(
					new pcl::PointCloud<PointT>);

			pcl::ExtractIndices<PointT> ex;
			ex.setInputCloud(planarCloud);
			ex.setIndices(*it);
			ex.filter(*segment_cloud);

			extractCollisions(cluster,segment_cloud);
		}
		return true;
	}

};

