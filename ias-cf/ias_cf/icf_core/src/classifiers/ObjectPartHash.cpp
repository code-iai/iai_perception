//WORK IN PROGRESS

/*

* ObjectPartHash.cpp
 *
 *  Created on: Feb 17, 2011
 *      Author: ferka
 */
#include "icf_core/classifiers/ObjectPartHash.h"

using namespace std;
using namespace icf;

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

	virtual ~ObjectPartExamples();

	/**
	 * \brief search for matching descriptors
	 * \param descriptor
	 * \return list of ID for which descriptor matches
	 */
	std::vector<std::pair<int, double> > search(const Eigen::VectorXf &descriptor)
			{
		std::vector<std::pair<int, double> > result;

		std::multimap<float,Eigen::VectorXf>::const_iterator end = descriptorList.end();//lower_bound(descriptor.sum());
		std::multimap<float,Eigen::VectorXf>::const_iterator it_descr = descriptorList.begin();//lower_bound(descriptor.sum()*0.9);
		double squared_gausian_parameter_inside = 100;
		double squared_gausian_parameter_outside = 100;
		double max = 0;
		double min_inside = DBL_MAX;
		double min_outside = DBL_MAX;

		for(;it_descr!=end;++it_descr)
		{
			//std::cerr<<"ITT-e??";
			double weight = exp( -jeffriesDistance(descriptor, it_descr->second)/squared_gausian_parameter_outside );
			if(max <  weight)
				max = weight;

			//distance = euclidianDistance(descriptor, it_descr->second);
			//if(min > distance)
			// min = distance;
		}
		/*for(;it_descr!=descriptorList.end();++it_descr)
	  {
		bool inside = true;
	    for(int j=0;j<descriptor.size();j++)
	    {
	      if( descriptor[j] > it_descr->second[j] )
	      {
	        inside=false;
	        break;
	      }
	    }
	    if(inside)
	    {
	    	//std::cerr<<"inside"<<std::endl;
	      double weight = exp( -euclidianDistance(descriptor, it_descr->second)/squared_gausian_parameter_inside );
	      if(max <  weight)
	        max = weight;

	      distance = euclidianDistance(descriptor, it_descr->second)/squared_gausian_parameter_outside;
	      if(min > distance)
	        min = distance;
	    }
	    else
	    {
	    	//std::cerr<<"outside"<<std::endl;
	      double weight = exp( -euclidianDistance(descriptor, it_descr->second)/squared_gausian_parameter_outside );
	      if(max <  weight)
	        max = weight;

	      distance = euclidianDistance(descriptor, it_descr->second)/squared_gausian_parameter_outside;
	      if(min > distance)
	        min = distance;
	    }
	  }*/
		//result.push_back(std::pair<int, double>(ID,  exp( -(descriptor - it_descr->second).cwise().abs().sum()/squared_gausian_parameter ) ) );
		/*{
	    bool match = true;
	    for(int j=0;j<descriptor.size();j++)
	    {

	      float upper_threshold = std::max(it_descr->second[j]*exp(0)*1.1, (double)it_descr->second[j]+10);
	      float lower_threshold = std::min(it_descr->second[j]*exp(0)*0.7, (double)it_descr->second[j]-10);
	      if( descriptor[j] > upper_threshold || descriptor[j] < lower_threshold )
	      {
	        //std::cout<<it_descr->second[j]*exp(0)*1.1<<"<"<<descriptor[j]<<std::endl;
	        match=false;
	        break;
	      }
	    }
	    if(match)
	      result.push_back(std::pair<int, double>(ID, 1.0));
	  }*/
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



void ObjectPartHash::addTrainingData(std::string data)
{
	this->trainingData = data;
}

void ObjectPartHash::addData(std::string data)
{
	this->classificationData = data;
}

void ObjectPartHash::addEvaluationData(std::string evaluationData)
{
	this->evaluationData = evaluationData;
}
