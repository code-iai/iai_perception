# launch classifier manager separately
#roslaunch icf_core icf_service_node.launch

# create classifier (its ID will be 0)
rosservice call /ias_classifier_manager/add_new_classifier -- "knn" "-m L2 -k 1"

# upload training data which has feature and label matrices
rosrun icf_core upload_dataset -m ias_classifier_manager -i $(rospack find icf_core)/data/test.h5 -n dataset1 -f /train -l /train_labels
# assign uploaded data to the "train" slot of the classifier
rosservice call /ias_classifier_manager/set_dataset -- "0" "train" "dataset1"
# build the model (knn gets the parameters at creation)
rosservice call /ias_classifier_manager/build_model -- "0" ""

# upload evaluation data which has feature and label matrices
rosrun icf_core upload_dataset -m ias_classifier_manager -i $(rospack find icf_core)/data/test.h5 -n dataset2 -f /test -l /test_labels
# assign uploaded data to the "eval" slot of the classifier
rosservice call /ias_classifier_manager/set_dataset -- "0" "eval" "dataset2"
# evaluate classifier
rosservice call /ias_classifier_manager/evaluate -- "0" # TODO: print success rate on ROS_INFO?
# TODO make it print on ROS_INFO or create tool using client: rosservice call /ias_classifier_manager/get_conf_matrix -- "0"

# upload test data which has features (labels not used)
rosrun icf_core upload_dataset -m ias_classifier_manager -i $(rospack find icf_core)/data/test.h5 -n dataset3 -f /blub -l /blub # TODO: better solution?
# assign uploaded data to the "eval" slot of the classifier
rosservice call /ias_classifier_manager/set_dataset -- "0" "classify" "dataset3" # TODO: what error?
# classify data
rosservice call /ias_classifier_manager/classify -- "0"

rosrun icf_core build_classifier -c knn -t vfh-complete-vosch.h5 -e vfh-test-vosch.h5 -n vfh -o vfh_vosch_knn
