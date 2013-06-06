/*
 * Copyright (c) 2012,
 * Zoltan-Csaba Marton <marton@cs.tum.edu>,
 * Ferenc Balint-Benczedi <balintb.ferenc@gmail.com>,
 * Florian Seidel <seidel.florian@gmail.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Intelligent Autonomous Systems Group/
 *       Technische Universitaet Muenchen nor the names of its contributors
 *       may be used to endorse or promote products derived from this software
 *       without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * STANDARD INCLUDES
 */
#include <map>
#include <string>

/*
 * ROS RELATED
 */
#include "ros/ros.h"

/*
 * MESSAGES
 */

#include <icf_core/NewClassifier.h>
#include <icf_core/BuildModel.h>
#include <icf_core/TrainData.h>
#include <icf_core/TestData.h>
#include <icf_core/Evaluate.h>
#include <icf_core/GetConfMatrix.h>
#include <icf_core/Classify.h>
#include <icf_core/EvalData.h>
#include <icf_core/Free.h>
#include <icf_core/Load.h>
#include <icf_core/Save.h>
#include <icf_core/AddDataset.h>
#include <icf_core/RemoveDataset.h>
#include <icf_core/SetDataset.h>
#include <icf_core/ReadDataset.h>
#include <icf_core/ClearConfMatrix.h>
#include <icf_core/ExecuteService.h>

/*
 * BASE
 */

#include "icf_dataset/DataSet.hpp"
#include "icf_core/base/ICFExceptionErrors.h"
#include "icf_core/base/Classifier.hpp"

/*
 * CLASSIFIERS
 */
#include <icf_core/classifiers/BoostOAA.h>
#include <icf_core/classifiers/SVMClassifier.h>
#include <icf_core/classifiers/ObjectPartHash.h>
#include <icf_core/service/knninst.h>


#ifndef CLASSIFIERMANAGER_H_
#define CLASSIFIERMANAGER_H_

#define SERVICE_NAME_ADD_NEW_CLASSIFIER "add_new_classifier"
#define SERVICE_NAME_BUILD_MODEL  "build_model"
#define SERVICE_NAME_ADD_TRAINING_DATA  "add_training_data"
#define SERVICE_NAME_SET_DATA "set_data"
#define SERVICE_NAME_GET_CONF_MATRIX "get_conf_matrix"
#define SERVICE_NAME_EVALUATE  "evaluate"
#define SERVICE_NAME_SET_EVALUATION_DATA  "set_evaluation_data"
#define SERVICE_NAME_CLASSIFY  "classify"
#define SERVICE_NAME_SAVE  "save"
#define SERVICE_NAME_LOAD  "load"
#define SERVICE_NAME_LOAD_AND_CLASSIFY  "load_and_classify"
#define SERVICE_NAME_LOAD_AND_EVALUATE "load_and_evaluate"
#define SERVICE_NAME_EXECUTE_SERVICE "executeService"
#define SERVICE_NAME_ADD_DATASET "add_dataset"
#define SERVICE_NAME_REMOVE_DATASET "remove_dataset"
#define SERVICE_NAME_SET_DATASET "set_dataset"
#define SERVICE_NAME_READ_DATASET "read_dataset"
#define SERVICE_NAME_CLEAR_CONF_MATRIX "clear_conf_matrix"
#define SERVICE_NAME_FREE "free"

namespace icf
{

class ClassifierManager
{
public:
  ClassifierManager(ros::NodeHandle n);
  ~ClassifierManager();
  ros::NodeHandle n;
  std::string name;
  ros::ServiceServer addNewClassifierService;
  ros::ServiceServer buildModelService;
  ros::ServiceServer addTrainingDataService;
  ros::ServiceServer setDataService;
  ros::ServiceServer getConfMatrixService;
  ros::ServiceServer evaluateService;
  ros::ServiceServer setEvalDataService;
  ros::ServiceServer classifyService;
  ros::ServiceServer loadService;
  ros::ServiceServer saveService;
  ros::ServiceServer loadAndClassifyService;
  ros::ServiceServer loadAndEvaluateService;
  ros::ServiceServer executeServiceService;
  ros::ServiceServer addDatasetService;
  ros::ServiceServer removeDatasetService;
  ros::ServiceServer setDatasetService;
  ros::ServiceServer readDatasetService;
  ros::ServiceServer clearConfMatrixService;
  ros::ServiceServer freeService;
  //std::vector<Classifier*> classifier_collection;
  std::map<int, Classifier*> classifier_collection;
  int nextClassifierId;
  std::map<std::string, DataSet<double>::Ptr> datasets;

  /**
   *\brief create a new classifier
   *\param request param
   *\param response
   */
  bool addNewClassifier(icf_core::NewClassifier::Request &req, icf_core::NewClassifier::Response &res);

  /**
   *\brief atm. returns as a response the maxID from a classifier specified in the request
   *\param request param
   *\param response
   */
  bool buildModel(icf_core::BuildModel::Request &req, icf_core::BuildModel::Response &res);

  /**
   *\brief adds Training data to a specified classifier
   *\param request param
   *\param response
   */
  bool addTrainData(icf_core::TrainData::Request &req, icf_core::TrainData::Response &res);

  /**
   *\brief runs classify for test data for a specified classifier
   *\param request param
   *\param response
   */
  bool addTestData(icf_core::TestData::Request &req, icf_core::TestData::Response &res);

  bool readDataset(icf_core::ReadDataset::Request &req, icf_core::ReadDataset::Response &res);

  bool getConfusionMatrix(icf_core::GetConfMatrix::Request &req, icf_core::GetConfMatrix::Response &res);

  bool evaluateTest(icf_core::Evaluate::Request &req, icf_core::Evaluate::Response &res);

  bool classify(icf_core::Classify::Request & req, icf_core::Classify::Response & res);

  bool addEvaluationData(icf_core::EvalData::Request & req, icf_core::EvalData::Response& res);

  bool save(icf_core::Save::Request & req, icf_core::Save::Response& res);

  bool load(icf_core::Load::Request& req, icf_core::Load::Response& res);

  bool executeService(icf_core::ExecuteService::Request& req, icf_core::ExecuteService::Response& res);

  bool assignDataset(icf_core::SetDataset::Request &req, icf_core::SetDataset::Response &res);

  bool addDataset(icf_core::AddDataset::Request &req, icf_core::AddDataset::Response &res);

  bool removeDataset(icf_core::RemoveDataset::Request &req, icf_core::RemoveDataset::Response &res);

  bool clearConfMatrix(icf_core::ClearConfMatrix::Request &req, icf_core::ClearConfMatrix::Response &res);

  bool free(icf_core::Free::Request & req, icf_core::Free::Response & res);

private:
  bool checkID(int id) const;

};
}
#endif /* CLASSIFIERMANAGER_H_ */
