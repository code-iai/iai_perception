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

/** \brief */
class ObjectPartHash : public Classifier {
public:

	/*
	 * \brief constructor
	 * \param parameters unrolled in a string
	 */

	ObjectPartHash(std::string param)/*: dataMap(),*//*class_time(0),cl_nr(0),result_string()*/{
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
	virtual void addEvaluationData(std::string data);

	virtual int buildModel(std::string data){}

	virtual std::string classify(){}

	virtual void load(std::string filename){}

	virtual void save(std::string filename){}

	std::string evaluate(){};

	// \brief classifier type in after hashing
	int cl_type;
	std::string evaluationData;
	std::string classificationData;
	std::string trainingData;

};
}
#endif /* DATASTRUCT_H_ */
