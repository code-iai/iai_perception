# Clear files
rm *_train.hdf5
rm *_test.hdf5

# Estimate VFH features for the training and testing data
rosrun icf_feature_extraction extract_pcl_features --normals -e normals.pcd -b data/training/ -f vfh -o vfh_train.hdf5 -n x
rosrun icf_feature_extraction extract_pcl_features --normals -e normals.pcd -b data/testing/ -f vfh -o vfh_test.hdf5 -n x

#rosrun icf_feature_extraction extract_pcl_features --normals -e normals.pcd -b data/training/ -f grsd2 -r 0.01 -d 0.01 -o grsd_train.hdf5 -n x
#rosrun icf_feature_extraction extract_pcl_features --normals -e normals.pcd -b data/testing/ -f grsd2 -r 0.01 -d 0.01 -o grsd_test.hdf5 -n x

# Create datasets containing the corresponding labels, based on folder names
rosrun icf_feature_extraction extract_pcl_features --normals -e normals.pcd -b data/training/ -f labels -o labels_train.hdf5 -n y -l
rosrun icf_feature_extraction extract_pcl_features --normals -e normals.pcd -b data/testing/ -f labels -o labels_test.hdf5 -n y -l

# Merge labels into the dataset containing the features
rosrun icf_dataset join_datasets -a labels_train.hdf5 -b vfh_train.hdf5
rosrun icf_dataset join_datasets -a labels_test.hdf5 -b vfh_test.hdf5
#rosrun icf_dataset join_datasets -a labels_train.hdf5 -b grsd_train.hdf5 -s /y -t /y
#rosrun icf_dataset join_datasets -a labels_test.hdf5 -b grsd_test.hdf5 -s /y -t /y

# Simple test script in octave to print out contents of hdf5 files
octave $(rospack find icf_dataset)/scripts/read_hdf5_octave.m vfh_train.hdf5
octave $(rospack find icf_dataset)/scripts/read_hdf5_octave.m vfh_test.hdf5

# Launch classifier manager separately
#roslaunch icf_core icf_service_node.launch

# Create classifier and parse out its ID
ID=`rosservice call /ias_classifier_manager/add_new_classifier -- "svm", "" | awk -F " " '{print $2; exit}'`
echo "created classifier with ID $ID"

# Train SVM classifier (see LIBSVM)
rosrun icf_core upload_dataset -i vfh_train.hdf5 -n dataset1
rosservice call /ias_classifier_manager/set_dataset -- "$ID" "train" "dataset1"
rosservice call /ias_classifier_manager/build_model -- "$ID" "-a 2 -t 2 -v 5 -5:2:15 -15:2:3"

# Evaluate classifier
rosrun icf_core upload_dataset -i vfh_test.hdf5 -n dataset2
rosservice call /ias_classifier_manager/set_dataset -- "$ID" "eval" "dataset2"
rosservice call /ias_classifier_manager/evaluate -- "$ID"
