///WORK IN PROGESS

/*
 * ObjectPartHash.h
 *
 *  Created on: Feb 17, 2011
 *      Author: ferka
 */
#include <ros/ros.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <map>
#include <float.h>
#include <numeric>
#include <math.h>

#include <Eigen/Core>

#include <boost/algorithm/string.hpp>

#include <icf_core/base/Classifier.hpp>
#include <icf_dataset/DataSet.hpp>
#include <icf_core/classifiers/SVMClassifier.h>
#include <icf_core/classifiers/KNNClassifier.h>

#include <flann/flann.h>
#include <flann/general.h>


#ifndef DATASTRUCT_H_
#define DATASTRUCT_H_

#define c_NN 0
#define c_kNN 1
#define c_SVM 2

namespace icf
{

class ObjectPartExamples {
public:
	ObjectPartExamples()
	{
		// TODO Auto-generated constructor stub
		ID = 0;
		descriptorList.clear();
	}

	/** \brief  ID number*/
	int ID;

	/** \brief  map of pairs: sum of descriptor elements and descriptor*/
	std::multimap<float , Eigen::VectorXf> descriptorList;

	virtual ~ObjectPartExamples(){}

	/**
	 * \brief search for matching descriptors
	 * \param descriptor
	 * \return list of ID for which descriptor matches
	 */
	std::vector <std::pair < int, double > > search(const Eigen::VectorXf &descriptor)
	{
		std::vector<std::pair<int, double> > result;
		std::multimap<float,Eigen::VectorXf>::const_iterator end = descriptorList.end();//lower_bound(descriptor.sum());
		std::multimap<float,Eigen::VectorXf>::const_iterator it_descr = descriptorList.begin();//lower_bound(descriptor.sum()*0.9);
//		double squared_gausian_parameter_inside = 100;
		double squared_gausian_parameter_outside = 100;
		double max = 0;
//		double min_inside = DBL_MAX;
//		double min_outside = DBL_MAX;

		for(;it_descr!=end;++it_descr)
		{
			double weight = exp( -jeffriesDistance(descriptor, it_descr->second)/squared_gausian_parameter_outside );
			if(max <  weight)
				max = weight;
		}
		result.push_back(std::pair<int, double>(ID, max ));
		return result;
	}


	inline double euclidianDistance(const Eigen::VectorXf &v1, const Eigen::VectorXf &v2)
	{
		return (v1-v2).norm();
	}
	inline double manhattanDistance(const Eigen::VectorXf &v1, const Eigen::VectorXf &v2)
	{

		return (v1-v2).cwiseAbs().sum();
	}

	inline double jeffriesDistance(const Eigen::VectorXf &v1, const Eigen::VectorXf &v2)
	{
		return std::sqrt((v1.array().sqrt()-v2.array().sqrt()).square().sum());
	}

	inline double bhattacharyyaDistance(const Eigen::VectorXf &v1, const Eigen::VectorXf &v2)
	{
		double sum = 0;
		for(int i=0;i<v1.rows();i++)
		{
			sum += sqrt(abs(v1[i]-v2[i]));
			//  cerr<<"i="<<i<<"sum: "<<sum<<std::endl;
		}
		std::cerr<<"returning: "<<-log(sum);
		return -log (sum) ;
	}
	inline double chiSquareDistance(const Eigen::VectorXf &v1, const Eigen::VectorXf &v2)
	{
		double sum=0;
		for(int i=0;i<v1.rows();i++)
		{
			if(v1[i]+v2[i] == 0)
				continue;
			sum +=(double)((v1[i]-v2[i])*(v1[i]-v2[i]))/(v1[i]+v2[i]);
			//cerr<<"i="<<i<<"sum: "<<sum<<std::endl;
		}
		return sum;
	}
	inline double klDivergenceDistance(const Eigen::VectorXf &v1, const Eigen::VectorXf &v2)
	{
		double sum=0;
		for(int i=0;i<v1.rows();i++)
		{
			if(v2[i]==0)
				continue;
			sum += (v1[i]-v2[i])*log(v1[i]/v2[i]);
		}
		return sum;
	}
};
class Arrangement {
public:
	Arrangement():classifier(NULL) {
		// TODO Auto-generated constructor stub
		arrangement_key = 0;
		objects.clear();
	}
	~Arrangement() {
		// TODO Auto-generated destructor stub
		/* if(classifier!=NULL)*/
		if(classifier)
			delete classifier;
	}

	/** \brief Searches for matching descriptors with the ID-s in the list possible
	 * \param possible the set of possible ID-s
	 * \param descriptor the descriptor to search for
	 * \return the ID-s of the matching entries
	 */
	//std::vector<std::pair<int, double> > search(const std::set<int> &possible, const Eigen::VectorXf &descriptor, int cl_type);
	Classifier *classifier;
	/** \brief the arrangement key*/
	int arrangement_key;
	std::vector<int> ID_counter;

	/** \brief list of ObjectPartExamples(object ID and it's list of descriptors ) */
	std::vector<ObjectPartExamples> objects;
	icf::DS data;
	flann::Matrix<float> knn_data;

	std::vector<std::pair<int, double> > search(const std::set<int> &possible, const Eigen::VectorXf &descriptor, int cl_type){
		//TODO return result as reference
		std::vector<std::pair<int, double> > result;
		switch(cl_type){
		case 0:{//NN
			for (std::vector<ObjectPartExamples>::iterator i=objects.begin();i!=objects.end();++i)
			{
				//set <int> :: iterator findIter;
				//findIter = possible.find(i->ID);
				//if(findIter!=possible.end()){
				std::vector<std::pair<int, double> > temp = i->search(descriptor);
				result.insert(result.end(), temp.begin(), temp.end());
				//}
			}
			break;
		}
		case 1:{//kNN
			/*			icf::DS test_data;
			DS::MatrixPtr m1 = test_data.addFeatureMatrix(1,descriptor.rows(),"x");
			for(int i=0;i<m1->cols();++i )
				(*m1)(0,i) = descriptor(i);
			classifier->setDataset(test_data, "classify");

			std::string res = classifier->classify();
			icf::ClassificationResult cr(res);
			//std::cerr<<classifier->classificationResult->confidences.get();
			std::cerr<<*cr.confidences;
			//cr.confidenceFor(0,2);
			std::cerr<<"RESULT:"<<std::endl<<res<<std::endl;
			std::cerr<<"--------------"<<std::endl;*/
			break;
		}
		case 2:{//SVM
			break;
		}
		default:{
			throw std::string("Invalid classifier exception");
			break;
		}
		}
		return result;
	}//function end*/


};
/** \brief */
class ObjectPartHash : public Classifier {
public:

	/*
	 * \brief constructor
	 * \param parameters unrolled in a string
	 */

	ObjectPartHash(std::string param): dataMap(),max_ID(-1)/*,//*class_time(0),cl_nr(0),result_string()*/{
		//max_ID = -1;
		cl_type = atoi(param.c_str());
		std::cerr<<"[note] Classifier based on Hashing created, with";
		if(cl_type == 0)
			std::cerr<<"NN ";
		else if(cl_type == 1)
			std::cerr<<"kNN ";
		else if(cl_type == 2)
			std::cerr<<"SVM ";
		std::cerr<<std::endl;
		//built =false;
	}

	virtual ~ObjectPartHash(){}

	/**
	 *\brief method for adding training data
	 *\param data to be interperted and added; cell separator: space; row seperator: \n
	 *\param  each row one example; last column label;
	 */
	virtual void addTrainingData(std::string data);

	/**
	 *\brief method for adding data
	 *\param data to be interperted and added; cell separator: space; row seperator: \n
	 */
	virtual void addData(std::string data);

	/**
	 * \brief method for adding evaluation data (use this only if you have ground truth)
	 * \param data
	 */
	virtual void addEvaluationData(std::string data){}

	virtual int buildModel(std::string data);

	virtual std::string classify();

	virtual void load(std::string filename);

	virtual void save(std::string filename){};

	std::string evaluate(){};

	void addToHashTable (int partNr, int arrangement_key, int ID, Eigen::VectorXf descriptor);

	std::map<int, std::vector<double> >  vote(std::string , std::vector<int> &);

	int addTrainingExample (std::string);

	void printData();

	void splitNonEmpty(std::vector<std::string> &values,  std::string input,std::string is_any_of)
	{
		boost::split (values, input, boost::is_any_of (is_any_of), boost::token_compress_on);
		std::vector<std::string>::iterator it = values.begin ();
		while (it != values.end())
		{
			if (*it == "")
				it = values.erase(it);
			else
				it++;
		}
	}
	 double getResultWeight(int partNr)
	  {
	    return std::sqrt(std::sqrt(partNr));
	  }


	double my_clock()
	{
		struct timeval tv;
		gettimeofday(&tv, NULL);
		return tv.tv_sec + (double)tv.tv_usec*1e-6;
	}
	// \brief classifier type in after hashing
	int cl_type;

	std::map<int, std::vector<Arrangement> > dataMap;

	int max_ID;

//	std::string evaluationData;
//	std::string classificationData;
//	std::string trainingData;

	LabelMap * labelMap;

	std::stringstream result_string;

};
}
#endif /* DATASTRUCT_H_ */
