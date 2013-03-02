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

#include <icf_core/service/ClassifierManager.h>

using namespace icf;
using namespace std;

ClassifierManager::ClassifierManager(ros::NodeHandle nh) :
    				nextClassifierId(0)
{
	n = nh;
	addNewClassifierService = n.advertiseService(SERVICE_NAME_ADD_NEW_CLASSIFIER, &ClassifierManager::addNewClassifier,
			this);
	buildModelService = n.advertiseService(SERVICE_NAME_BUILD_MODEL, &ClassifierManager::buildModel, this);
	addTrainingDataService = n.advertiseService(SERVICE_NAME_ADD_TRAINING_DATA, &ClassifierManager::addTrainData, this);
	setDataService = n.advertiseService(SERVICE_NAME_SET_DATA, &ClassifierManager::addTestData, this);
	getConfMatrixService = n.advertiseService(SERVICE_NAME_GET_CONF_MATRIX, &ClassifierManager::getConfusionMatrix, this);
	evaluateService = n.advertiseService(SERVICE_NAME_EVALUATE, &ClassifierManager::evaluateTest, this);
	setEvalDataService = n.advertiseService(SERVICE_NAME_SET_EVALUATION_DATA, &ClassifierManager::addEvaluationData,
			this);
	classifyService = n.advertiseService(SERVICE_NAME_CLASSIFY, &ClassifierManager::classify, this);
	saveService = n.advertiseService(SERVICE_NAME_SAVE, &ClassifierManager::save, this);
	loadService = n.advertiseService(SERVICE_NAME_LOAD, &ClassifierManager::load, this);
	executeServiceService = n.advertiseService(SERVICE_NAME_EXECUTE_SERVICE, &ClassifierManager::executeService, this);
	addDatasetService = n.advertiseService(SERVICE_NAME_ADD_DATASET, &ClassifierManager::addDataset, this);
	removeDatasetService = n.advertiseService(SERVICE_NAME_REMOVE_DATASET, &ClassifierManager::removeDataset, this);
	setDatasetService = n.advertiseService(SERVICE_NAME_SET_DATASET, &ClassifierManager::assignDataset, this);
	readDatasetService = n.advertiseService(SERVICE_NAME_READ_DATASET, &ClassifierManager::readDataset, this);
	clearConfMatrixService = n.advertiseService(SERVICE_NAME_CLEAR_CONF_MATRIX, &ClassifierManager::clearConfMatrix,
			this);
	freeService = n.advertiseService(SERVICE_NAME_FREE, &ClassifierManager::free, this);
	ROS_INFO("Ready to receive");
}

bool ClassifierManager::save(icf_core::Save::Request & req, icf_core::Save::Response& res)
{
	ROS_DEBUG("Saving. Classifier ID: %ld", (long int)req.ID);
	try
	{
		if (!checkID(req.ID))
		{
			res.result = "invalid ID";
			ROS_ERROR("INVALID CLASSIFIER ID!");
			return false;
		}
		try
		{
			classifier_collection[req.ID]->save(req.filename);
		}
		catch (std::string msg)
		{
			res.result = msg;
			return false;
		}
		res.result = "saved";
		return true;
	}
	catch (boost::exception& e)
	{
		res.result = boost::diagnostic_information(e);
		return false;
	}
	catch (...)
	{
		res.result = "Unhandled exception was thrown...";
		return false;
	}
}

bool ClassifierManager::readDataset(icf_core::ReadDataset::Request &req, icf_core::ReadDataset::Response &res)
{
	try
	{
		fstream in(req.filePath.c_str(), ios_base::in | ios_base::binary);
		DS::Ptr dataset(new DS(req.filePath));
		this->datasets[req.ID] = dataset;
		res.result = "Added dataset";
		return true;
	}
	catch (boost::exception& e)
	{
		res.result = boost::diagnostic_information(e);
		return false;
	}
	catch (...)
	{
		res.result = "Uncaught exception";
		return false;
	}
	return true;
}

bool ClassifierManager::checkID(int id) const
{
	return classifier_collection.find(id) != classifier_collection.end();
}

bool ClassifierManager::load(icf_core::Load::Request& req, icf_core::Load::Response& res)
{
	std::cerr<<"LOADING INTO CLASSIFIER!!"<<std::endl;
	try
	{
		if (!checkID(req.ID))
		{
			std::cerr<<"NO SUCH ID!!1"<<std::endl;
			res.result = "invalid ID";
			return false;
		}
		try
		{
			std::cerr<<"ID EXISTS NOW WHAT"<<std::endl;

			classifier_collection[req.ID]->load(req.filename);
			std::cerr<<"ID LOADED!!!"<<std::endl;
			res.result = "loaded classifier";
			if (classifier_collection[req.ID]->cm)
			{
				std::cerr<<"CONF MATRIX EXISTS!...LOADING!!!"<<std::endl;
				res.confMatrix = classifier_collection[req.ID]->cm->serialize();
			}
			return true;
		}
		catch (boost::exception& e)
		{
			res.result = boost::diagnostic_information(e);
			return false;
		}
		catch (std::string msg)
		{
			res.result = msg;
			return false;
		}
	}
	catch (...)
	{
		res.result = "Uncaught exception";
		return false;
	}
}

bool ClassifierManager::addNewClassifier(icf_core::NewClassifier::Request &req, icf_core::NewClassifier::Response &res)
{
	try
	{
		if (boost::starts_with(req.classifier_type, "oph"))
		{
			try
			{
				ObjectPartHash *oph = new ObjectPartHash(req.parameters);
				oph->type = "oph";
				res.ID = nextClassifierId;//classifier_collection.size();
				classifier_collection[nextClassifierId++] = ((Classifier*) oph);
			}
			catch (boost::exception& e)
			{
				res.result = boost::diagnostic_information(e);
				return false;
			}
			catch (std::string& e)
			{
				res.result = e;
				std::cerr << e << endl;
				return false;
			}
		}
		else if (boost::starts_with(req.classifier_type, "svm"))
		{
			try
			{
				SVMClassifier *classifier = new SVMClassifier(req.parameters);
				classifier->type = req.classifier_type;
				res.ID = nextClassifierId; //classifier_collection.size();
				classifier_collection[nextClassifierId++] = ((Classifier*)classifier);
			}
			catch (boost::exception& e)
			{
				res.result = boost::diagnostic_information(e);
				return false;
			}
			catch (std::string& msg)
			{
				res.result = msg;
				std::cerr << msg << endl;
				return false;
			}
		}
		else if (boost::starts_with(req.classifier_type, "knn"))
		{
			try
			{
				Classifier * classifier = knninst(req.parameters);
				classifier->type = req.classifier_type;
				res.ID = nextClassifierId; //classifier_collection.size();
				classifier_collection[nextClassifierId++] = (Classifier*)classifier;
			}
			catch (boost::exception& e)
			{
				res.result = boost::diagnostic_information(e);
				return false;
			}
			catch (std::invalid_argument& e)
			{
				std::cerr << e.what() << endl;
				res.result = e.what();
				return false;
			}
			catch (std::string& e)
			{
				res.result = e;
				std::cerr << e << endl;
				return false;
			}

		}
		else if (boost::starts_with(req.classifier_type, "boost"))

		{
			try
			{
				BoostOAA *classifier = new BoostOAA(req.parameters);
				classifier->type = req.classifier_type;
				res.ID = nextClassifierId; //classifier_collection.size();
				classifier_collection[nextClassifierId++] = (Classifier*)classifier;
			}
			catch (boost::exception& e)
			{
				res.result = boost::diagnostic_information(e);
				return false;
			}
			catch (std::string msg)
			{
				res.result = msg;
				std::cerr << msg << endl;
				return false;
			}
		}
		else //trying to add a classifer that is not implemented
		{
			//TODO: throw or something
			res.result = "Invalid classifier name";
			res.ID = -1;
			return false;
		}
		return true;
	}
	catch (...)
	{
		res.result = "Uncaught exception.";
		return false;
	}
}

bool ClassifierManager::buildModel(icf_core::BuildModel::Request &req, icf_core::BuildModel::Response &res)
{
	try
	{
		if (!checkID(req.ID))
		{
			res.result = "invalid ID";
			return false;
		}
		try
		{
			// TODO is this used somewhere? maybe use better name for it
			int nr_ID = classifier_collection[req.ID]->buildModel(req.parameters);
			if (nr_ID == 0)
			{
				std::ostringstream sb;
				sb << nr_ID;
				res.result = sb.str();
			}
			else
			{
				res.result = "could not build classifier";
			}
			return true;
		}
		catch (boost::exception& e)
		{
			res.result = boost::diagnostic_information(e);
			return false;
		}
		catch (std::string msg)
		{
			res.result = msg;
			std::cerr << msg << endl;
			return false;
		}
	}
	catch (...)
	{
		res.result = "Uncaught exception";
		return false;
	}
}

bool ClassifierManager::addTrainData(icf_core::TrainData::Request &req, icf_core::TrainData::Response &res)
{
	try
	{
		if (!checkID(req.ID))
		{
			res.result = "invalid ID";
			return false;
		}
		classifier_collection[req.ID]->addTrainingData(req.data);
		res.result = "Succesfuly added training data";
		return true;
	}
	catch (boost::exception& e)
	{
		res.result = boost::diagnostic_information(e);
		return false;
	}
	catch (std::string msg)
	{
		res.result = msg;
		return false;
	}
	catch (...)
	{
		res.result = "Uncaught exception";
		return false;
	}
}

bool ClassifierManager::assignDataset(icf_core::SetDataset::Request &req, icf_core::SetDataset::Response &res)
{
	int clID = req.clID;
	string clSlot = req.clSlot;
	string dsID = req.dsID;
	try
	{
		if (!checkID(clID))
		{
			res.result = "invalid ID";
			return false;
		}
		if (this->datasets.find(dsID) != this->datasets.end())
		{
			classifier_collection[clID]->setDataset(this->datasets[dsID], clSlot);
			res.result = "Added data";
			return true;
		}
		else
		{
			res.result = "No such dataset";
			return false;
		}
	}
	catch (boost::exception& e)
	{
		res.result = boost::diagnostic_information(e);
		return false;
	}
	catch (std::string msg)
	{
		return false;
	}
	catch (...)
	{
		res.result = "Uncaught exception";
		return false;
	}
}

bool ClassifierManager::addDataset(icf_core::AddDataset::Request &req, icf_core::AddDataset::Response &res)
{
	try
	{
		void* data = (void*)&req.data[0];
		size_t length = req.data.size();
		DS::Ptr ds(new DS(data, length, false));
		this->datasets[req.ID] = ds;
		res.result = "Added dataset ";
		return true;
	}
	catch (boost::exception& e)
	{
		res.result = boost::diagnostic_information(e);
		return false;
	}
	catch (...)
	{
		res.result = "Uncaught exception";
		return false;
	}
}

bool ClassifierManager::removeDataset(icf_core::RemoveDataset::Request &req, icf_core::RemoveDataset::Response &res)
{
	try
	{
		if (this->datasets.find(req.ID) != this->datasets.end())
		{
			this->datasets.erase(req.ID);
			res.result = "Removed dataset ";
			return true;
		}
		else
		{
			res.result = "Invalid dataset ID";
			return false;
		}
	}
	catch (boost::exception& e)
	{
		res.result = boost::diagnostic_information(e);
		return false;
	}
	catch (...)
	{
		res.result = "Uncaught exception";
		return false;
	}
}

bool ClassifierManager::addTestData(icf_core::TestData::Request &req, icf_core::TestData::Response &res)
{
	try
	{
		if (!checkID(req.ID))
		{
			res.result = "invalid ID";
			return false;
		}
		classifier_collection[req.ID]->addData(req.data);
		res.result = "Added data";
		return true;
	}
	catch (boost::exception& e)
	{
		res.result = boost::diagnostic_information(e);
		return false;
	}
	catch (std::string msg)
	{
		res.result = msg;
		return false;
	}
}

bool ClassifierManager::classify(icf_core::Classify::Request & req, icf_core::Classify::Response & res)
{
	try
	{
		if (!checkID(req.ID))
		{
			res.result = "invalid ID";
			return false;
		}
		res.result = classifier_collection[req.ID]->classify();
		return true;
	}
	catch (boost::exception& e)
	{
		res.result = boost::diagnostic_information(e);
		return false;
	}
	catch (...)
	{
		res.result = "Uncaught exception";
		return false;
	}
}

bool ClassifierManager::addEvaluationData(icf_core::EvalData::Request & req, icf_core::EvalData::Response& res)
{
	try
	{
		if (!checkID(req.ID))
		{
			res.result = "invalid ID";
			return false;
		}
		classifier_collection[req.ID]->addEvaluationData(req.data);
		res.result = "Added evaluation data";
		return true;
	}
	catch (boost::exception& e)
	{
		res.result = boost::diagnostic_information(e);
		return false;
	}
	catch (...)
	{
		res.result = "Uncaught exception";
		return false;
	}
}

bool ClassifierManager::evaluateTest(icf_core::Evaluate::Request &req, icf_core::Evaluate::Response &res)
{
	try
	{
		if (!checkID(req.ID))
		{
			res.result = "invalid ID";
			return false;
		}
		res.result = classifier_collection[req.ID]->evaluate();
		return true;
	}
	catch (boost::exception& e)
	{
		res.result = boost::diagnostic_information(e);
		return false;
	}
	catch (...)
	{
		res.result = "Uncaught exception";
		return false;
	}
}

bool ClassifierManager::executeService(icf_core::ExecuteService::Request& req, icf_core::ExecuteService::Response& res)
{
	try
	{
		if (!checkID(req.ID))
		{
			res.result = "invalid ID";
			return false;
		}
		res.result = classifier_collection[req.ID]->executeService(req.service, req.params);
		return true;
	}
	catch (boost::exception& e)
	{
		res.result = boost::diagnostic_information(e);
		return false;
	}
	catch (...)
	{
		res.result = "Uncaught exception";
		return false;
	}
}

bool ClassifierManager::getConfusionMatrix(icf_core::GetConfMatrix::Request &req,
		icf_core::GetConfMatrix::Response &res)
{
	try
	{
		if (!checkID(req.ID))
		{
			res.result = "invalid ID";
			return false;
		}
		std::stringstream temp;
		/*if (classifier_collection[req.ID]->type == "oph")
     temp << classifier_collection[req.ID]->confusionMatrix;
     else*/
		temp << classifier_collection[req.ID]->cm;
		res.result = temp.str();
		return true;
	}
	catch (boost::exception& e)
	{
		res.result = boost::diagnostic_information(e);
		return false;
	}
	catch (...)
	{
		res.result = "Uncaught exception";
		return false;
	}
}

bool ClassifierManager::clearConfMatrix(icf_core::ClearConfMatrix::Request &req,
		icf_core::ClearConfMatrix::Response &res)
{
	try
	{
		//res.success = true;
		if (!checkID(req.ID))
		{
			res.result = "invalid ID";
			return true;
		}
		//std::stringstream temp;
		/*		if (classifier_collection[req.ID]->type == "oph")
     classifier_collection[req.ID]->confusionMatrix =
     Eigen::MatrixXf::Zero(
     classifier_collection[req.ID]->confusionMatrix.rows(),
     classifier_collection[req.ID]->confusionMatrix.cols());*/
		//else
		//classifier_collection[req.ID]->cm->
		res.result = "Cleared";
		return true;
	}
	catch (boost::exception& e)
	{
		res.result = boost::diagnostic_information(e);
		return false;
	}
	catch (...)
	{
		res.result = "Uncaught exception";
		return true;
	}
}

ClassifierManager::~ClassifierManager()
{
	std::map<int, Classifier*>::iterator it = classifier_collection.begin();
	for (; it != classifier_collection.end(); it++)
	{
		delete it->second;
	}
	classifier_collection.clear();
}

bool ClassifierManager::free(icf_core::Free::Request & req, icf_core::Free::Response & res)
{
	try
	{
		//res.success = true;
		if (!checkID(req.ID))
		{
			res.result = "invalid ID";
			return true;
		}
		Classifier * c = classifier_collection[req.ID];
		classifier_collection.erase(req.ID);
		delete c;
		res.result = "removed classifier";
		return true;
	}
	catch (boost::exception& e)
	{
		res.result = boost::diagnostic_information(e);
		return false;
	}
	catch (...)
	{
		res.result = "Uncaught exception";
		//res.success = false;
		return true;
	}
}
