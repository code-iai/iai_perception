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

#include <icf_core/client/Client.h>

namespace icf
{

#define INST_CLIENT(serviceClnt,serviceType,serviceName)\
		serviceClnt=nh.serviceClient<serviceType> (serviceName);\
		if(!(serviceClnt.exists() && serviceClnt.isValid())) \
		{\
			errors.push_back(service_unavailable_error(serviceName));\
		}

/*****************************************************************************
 * 									Client Base
 *****************************************************************************/

ClassifierBase::ClassifierBase() :
    invalidState(false)
{

}

ClassifierBase::~ClassifierBase()
{

}

ServiceCallResultHolder::ServiceCallResultHolder() :
    lastResponse("")
{

}

ServiceCallResultHolder::~ServiceCallResultHolder()
{

}

std::string ServiceCallResultHolder::getLastResponse()
{
  return lastResponse;
}

void ServiceCallResultHolder::setLastResponse(const std::string& lastResponse)
{
  this->lastResponse = lastResponse;
}

template<class S, class C>
  void callService(S client, C& call, const std::string& serviceName, ServiceCallResultHolder* base)
  {

    if (!client.isValid())
    {
      throw ICFException() << service_unavailable_error(serviceName);
    }

    if (client.call(call.request, call.response))
    {
      base->setLastResponse(call.response.result);
    }
    else
    {
      base->setLastResponse(call.response.result);
      throw ICFException()
          << service_call_returned_false_error(
              boost::make_tuple<std::string, std::string>(serviceName, call.response.result));
    }
  }

void ClassifierBase::checkState() const
{
  if (invalidState)
  {
    throw ICFException() << invalid_state_error("ClassifierClient in invalid state");
  }
}

/*****************************************************************************
 * 									ClassifierClient
 *****************************************************************************/

std::string ensureServiceBasePathFormat(const std::string& serviceBasePath)
{
  std::string copy(serviceBasePath);
  if (serviceBasePath.substr(serviceBasePath.size() - 1, 1) != "/")
    copy += "/";
  if (serviceBasePath.substr(0, 1) != "/")
    copy = "/" + copy;
  return copy;
}

ClassifierClient::ClassifierClient(ros::NodeHandle& nh, const std::string& serviceBasePath,
                                   boost::shared_ptr<DataRepository> repo) :
    dataRepository(repo), nh(nh), _serviceBasePath(ensureServiceBasePathFormat(serviceBasePath)), classifierName(""), parameters(
        ""), classifierId(-1), buildModelName(_serviceBasePath + SERVICE_NAME_BUILD_MODEL), evaluateName(
        _serviceBasePath + SERVICE_NAME_EVALUATE), classifyName(_serviceBasePath + SERVICE_NAME_CLASSIFY), setDatasetName(
        _serviceBasePath + SERVICE_NAME_SET_DATASET), saveClassifierName(_serviceBasePath + SERVICE_NAME_SAVE), addNewClassifierName(
        _serviceBasePath + SERVICE_NAME_ADD_NEW_CLASSIFIER), loadClassifierName(_serviceBasePath + SERVICE_NAME_LOAD), freeClassifierName(
        _serviceBasePath + SERVICE_NAME_FREE)
{
  createClients();
  if (!repo)
  {
    dataRepository.reset(new ServerSideRepo(nh, _serviceBasePath));
  }
  else
  {
    dataRepository = repo;
  }
}

ClassifierClient::ClassifierClient(ros::NodeHandle& nh, const std::string& serviceBasePath,
                                   const std::string& classifierName, const std::string& parameters,
                                   boost::shared_ptr<DataRepository> repo) :
    dataRepository(repo), nh(nh), _serviceBasePath(ensureServiceBasePathFormat(serviceBasePath)), classifierName(
        classifierName), parameters(parameters), classifierId(-1), buildModelName(
        _serviceBasePath + SERVICE_NAME_BUILD_MODEL), evaluateName(_serviceBasePath + SERVICE_NAME_EVALUATE), classifyName(
        _serviceBasePath + SERVICE_NAME_CLASSIFY), setDatasetName(_serviceBasePath + SERVICE_NAME_SET_DATASET), saveClassifierName(
        _serviceBasePath + SERVICE_NAME_SAVE), addNewClassifierName(_serviceBasePath + SERVICE_NAME_ADD_NEW_CLASSIFIER), loadClassifierName(
        _serviceBasePath + SERVICE_NAME_LOAD), freeClassifierName(_serviceBasePath + SERVICE_NAME_FREE)
{
  createClients();
  icf_core::NewClassifier newClassifier;
  newClassifier.request.classifier_type = classifierName;
  newClassifier.request.parameters = parameters;

  if (!addNewClassifierClient.isValid())
  {
    throw ICFException() << service_unavailable_error(addNewClassifierName);
  }

  if (addNewClassifierClient.call(newClassifier.request, newClassifier.response))
  {
    setLastResponse(newClassifier.response.result);
    classifierId = newClassifier.response.ID;
  }
  else
  {
    setLastResponse(newClassifier.response.result);
    invalidState = true;
    throw ICFException() << service_call_returned_false_error(addNewClassifierName);
  }
  if (!repo)
  {
    dataRepository.reset(new ServerSideRepo(nh, _serviceBasePath));
  }
  else
  {
    dataRepository = repo;
  }
}

ClassifierClient::ClassifierClient(ros::NodeHandle& nh, const std::string& serviceBasePath, int classifierId,
                                   boost::shared_ptr<DataRepository> repo) :
    dataRepository(repo), nh(nh), _serviceBasePath(ensureServiceBasePathFormat(serviceBasePath)), classifierName(""), parameters(
        ""), classifierId(classifierId), buildModelName(_serviceBasePath + SERVICE_NAME_BUILD_MODEL), evaluateName(
        _serviceBasePath + SERVICE_NAME_EVALUATE), classifyName(_serviceBasePath + SERVICE_NAME_CLASSIFY), setDatasetName(
        _serviceBasePath + SERVICE_NAME_SET_DATASET), saveClassifierName(_serviceBasePath + SERVICE_NAME_SAVE), addNewClassifierName(
        _serviceBasePath + SERVICE_NAME_ADD_NEW_CLASSIFIER), loadClassifierName(_serviceBasePath + SERVICE_NAME_LOAD), freeClassifierName(
        _serviceBasePath + SERVICE_NAME_FREE)
{
  createClients();
  if (!repo)
  {
    dataRepository.reset(new ServerSideRepo(nh, _serviceBasePath));
  }
  else
  {
    dataRepository = repo;
  }
}

ClassifierClient::~ClassifierClient()
{

}

void ClassifierClient::createClients()
{
  std::vector<service_unavailable_error> errors;

  INST_CLIENT(buildModelClient, icf_core::BuildModel, buildModelName);
  INST_CLIENT(evaluateClient, icf_core::Evaluate, evaluateName);
  INST_CLIENT(classifyClient, icf_core::Classify, classifyName);
  INST_CLIENT(setDatasetClient, icf_core::SetDataset, setDatasetName);
  INST_CLIENT(addNewClassifierClient, icf_core::NewClassifier, addNewClassifierName);
  INST_CLIENT(saveClassifierClient, icf_core::Save, saveClassifierName);
  INST_CLIENT(loadClassifierClient, icf_core::Load, loadClassifierName);
  INST_CLIENT(freeClassifierClient, icf_core::Free, freeClassifierName);

  if (!errors.empty())
  {
    invalidState = true;
    throw ICFException() << service_unavailable_collection(errors);
  }
}

void ClassifierClient::checkState() const
{
  ClassifierBase::checkState();
  if (classifierId == -1)
  {
    throw ICFException() << invalid_classifier_id_error(classifierId);
  }
}

void ClassifierClient::assignData(const std::string& name, DataSetSlot type)
{
  checkState();
  icf_core::SetDataset setDataset;
  setDataset.request.clID = classifierId;
  if (type == Eval)
  {
    setDataset.request.clSlot = "eval";
  }
  else if (type == Train)
  {
    setDataset.request.clSlot = "train";
  }
  else if (type == Classify)
  {
    setDataset.request.clSlot = "classify";
  }
  setDataset.request.dsID = name;
  callService(setDatasetClient, setDataset, setDatasetName, this);
}

void ClassifierClient::train(const std::string& params)
{
  checkState();
  icf_core::BuildModel buildModel;
  buildModel.request.ID = classifierId;
  buildModel.request.parameters = params;
  callService(buildModelClient, buildModel, buildModelName, this);
}

void ClassifierClient::train(const std::string& params, DS& dataset, const std::string& features, const std::string& gt)
{
  checkState();
  srand(clock());
  std::stringstream ssname;
  ssname << rand();
  std::string name = ssname.str();
  dataRepository->uploadData(dataset, name, features, gt);
  assignData(name, Train);
  train(params);
  dataRepository->removeData(name);
}

EvaluationResult ClassifierClient::evaluate()
{
  checkState();
  icf_core::Evaluate evaluate;
  evaluate.request.ID = classifierId;
  callService(evaluateClient, evaluate, evaluateName, this);
  EvaluationResult evaluationResult(evaluate.response.result);
  confusionMatrix = evaluationResult.getConfusionMatrix();
  return evaluationResult;
}

EvaluationResult ClassifierClient::evaluate(DS& dataset, const std::string& features, const std::string& gt)
{
  checkState();
  srand(clock());
  std::stringstream ssname;
  ssname << rand();
  std::string name = ssname.str();
  dataRepository->uploadData(dataset, name, features, gt);
  assignData(name, Eval);
  EvaluationResult evaluationResult = evaluate();
  dataRepository->removeData(name);
  return evaluationResult;
}

ClassificationResult ClassifierClient::classify()
{
  icf_core::Classify classify;
  classify.request.ID = classifierId;
  callService(classifyClient, classify, classifyName, this);
  ClassificationResult result(classify.response.result);
  return result;
}

ClassificationResult ClassifierClient::classify(DS& dataset, const std::string& features)
{
  checkState();
  srand(clock());
  std::stringstream ssname;
  ssname << rand();
  std::string name = ssname.str();
  dataRepository->uploadData(dataset, name, features, "");
  assignData(name, Classify);
  ClassificationResult result = classify();
  dataRepository->removeData(name);
  return result;
}

void ClassifierClient::save(const std::string& filename)
{
  checkState();
  icf_core::Save save;
  save.request.ID = classifierId;
  save.request.filename = filename;
  callService(saveClassifierClient, save, saveClassifierName, this);
}

void ClassifierClient::load(const std::string& filename)
{
  checkState();
  icf_core::Load load;
  load.request.ID = classifierId;
  load.request.filename = filename;
  callService(loadClassifierClient, load, loadClassifierName, this);
  //load.response should contain the confusion matrix
  if (load.response.confMatrix != "")
  {
    confusionMatrix = boost::shared_ptr<ConfusionMatrix>(new ConfusionMatrix());
    confusionMatrix->deserialize(load.response.confMatrix);
  }

}

int ClassifierClient::getClassifierId() const
{
  return classifierId;
}

void ClassifierClient::setClassifierId(int classifierId)
{
  this->classifierId = classifierId;
}

boost::shared_ptr<ConfusionMatrix> ClassifierClient::getConfusionMatrix()
{
  return confusionMatrix;
}

void ClassifierClient::free()
{
  checkState();
  if (this->classifierId == -1)
  {
    return;
  }
  icf_core::Free free;
  free.request.ID = this->classifierId;
  callService(freeClassifierClient, free, freeClassifierName, this);
}

/*****************************************************************************
 * 									DataRepository
 *****************************************************************************/

DataRepository::~DataRepository()
{

}

ServerSideRepo::ServerSideRepo(ros::NodeHandle& nh, const std::string& serviceBasePath) :
    nh(nh), _serviceBasePath(ensureServiceBasePathFormat(serviceBasePath)), addDatasetName(
        _serviceBasePath + SERVICE_NAME_ADD_DATASET), setDatasetName(_serviceBasePath + SERVICE_NAME_SET_DATASET), removeDatasetName(
        _serviceBasePath + SERVICE_NAME_REMOVE_DATASET), readDatasetName(_serviceBasePath + SERVICE_NAME_READ_DATASET), invalidState(
        false)
{
  createClients();
}

ServerSideRepo::~ServerSideRepo()
{

}

void ServerSideRepo::createClients()
{
  std::vector<service_unavailable_error> errors;
  INST_CLIENT(addDatasetClient, icf_core::AddDataset, addDatasetName);
  INST_CLIENT(setDatasetClient, icf_core::SetDataset, setDatasetName);
  INST_CLIENT(readDatasetClient, icf_core::ReadDataset, readDatasetName);
  INST_CLIENT(removeDatasetClient, icf_core::RemoveDataset, removeDatasetName);

  if (!errors.empty())
  {
    invalidState = true;
    throw ICFException() << service_unavailable_collection(errors);
  }

}

void ServerSideRepo::checkState() const
{
  if (invalidState)
  {
    throw ICFException() << invalid_state_error("ClassifierClient in invalid state");
  }
}

void ServerSideRepo::uploadData(const std::string& pathToDataset, const std::string& name)
{
  checkState();
  icf_core::ReadDataset readDataset;
  readDataset.request.ID = name;
  readDataset.request.filePath = pathToDataset;
  callService(readDatasetClient, readDataset, readDatasetName, this);
}

void ServerSideRepo::uploadData(DS& dataset, const std::string& name, const std::string& features,
                                const std::string& gt)
{
  checkState();
  if (features != "")
  {

    if (!dataset.contains(features))
    {
      throw ICFException() << dataset_matrix_not_found_error(features);
    }
    if (features != "/x")
      dataset.alias(features, "/x");
  }
  if (gt != "")
  {

    if (!dataset.contains(gt))
    {
      throw ICFException() << dataset_matrix_not_found_error(gt);
    }
    if (gt != "/y")
      dataset.alias(gt, "/y");
  }
  icf_core::AddDataset addDataset;
  addDataset.request.ID = name;
  addDataset.request.data << dataset;
  callService(addDatasetClient, addDataset, addDatasetName, this);
}

void ServerSideRepo::removeData(const std::string& name)
{
  checkState();
  icf_core::RemoveDataset removeDataset;
  removeDataset.request.ID = name;
  callService(removeDatasetClient, removeDataset, removeDatasetName, this);
}

}
