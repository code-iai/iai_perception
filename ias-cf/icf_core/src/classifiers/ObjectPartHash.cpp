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

//this is our core "build model"
void ObjectPartHash::addToHashTable(int part_nr, int arrangement_key, int ID ,Eigen::VectorXf descriptor){

	//std::cerr<<"ADDING: ARR_KEY: "<<arrangement_key<<std::endl;
	//std::cerr<<"ADDING: DESCRIPTOR: "<<descriptor.transpose()<<std::endl;

	ObjectPartExamples pairIdDescription;
	pairIdDescription.ID= ID;
	if (ID > max_ID)
	{
		max_ID=ID;
		//    confusionMatrix = Eigen::MatrixXf::Zero(max_ID+1, max_ID+1);
		//    result_matrix = Eigen::MatrixXf::Zero(max_ID+1, max_ID+1);
		//    tpr = Eigen::VectorXi::Zero(max_ID+2);
	}
	float f = descriptor.sum();//
	pairIdDescription.descriptorList.insert(std::pair<float, Eigen::VectorXf>(f,descriptor));

	//IF ARRANGEMENT ALLREADY EXISTS IN MAP, WRITE DATA TO EXISTING ARRANGEMENT

	bool ArrangementExists = false;
	bool ObjectPartExampleExists = false;//

	std::map<int, std::vector<Arrangement> >::iterator it_Map = dataMap.lower_bound(part_nr);
	std::vector<Arrangement>::iterator it_Arrangement;
	unsigned int it;
	if(it_Map!=dataMap.end())
	{
		it_Arrangement = dataMap[part_nr].begin();

		for(;it_Arrangement < dataMap[part_nr].end(); ++it_Arrangement)
		{
			if(it_Arrangement->arrangement_key == arrangement_key)
			{
				ArrangementExists = true;
				//it_Arrangement->objects[it].ID;
				for(it = 0; it<it_Arrangement->objects.size()&&ArrangementExists;++it)
				{
					if(it_Arrangement->objects[it].ID == ID)
					{
						ObjectPartExampleExists = true;
						break;
					}
				}
				break;
			}
		}
	}
	if(ArrangementExists && ObjectPartExampleExists)
	{
		it_Arrangement->objects[it].descriptorList.insert(std::pair<float, Eigen::VectorXf>(f,descriptor));;
		it_Arrangement->ID_counter.resize(max_ID+1);
		it_Arrangement->ID_counter[ID]++;
	}
	else if(ArrangementExists && !ObjectPartExampleExists)
	{
		it_Arrangement->objects.push_back(pairIdDescription);
		it_Arrangement->ID_counter.resize(max_ID+1);
		it_Arrangement->ID_counter[ID]++;
	}

	// ELSE CREATE A NEW ARRANGEMENT AND ADD IT TO THE MAP
	else
	{
		//std::vector<Arrangement> arrangementKey_objectList_v;
		Arrangement arrangementKey_objectList;
		//icf::DS::Matrix& m = arrangementKey_objectList.data.addFeatureMatrix(100,10,"a");
		arrangementKey_objectList.arrangement_key = arrangement_key;
		arrangementKey_objectList.objects.push_back(pairIdDescription);
		arrangementKey_objectList.ID_counter.resize(max_ID+1);
		arrangementKey_objectList.ID_counter[ID]++;
		std::stringstream ststr_arr;
		ststr_arr<<arrangement_key;
		arrangementKey_objectList.data.name = ststr_arr.str();
		std::cerr<<"Arrangement Created:"<<arrangement_key<<"!!!!"<<std::endl;
		//arrangementKey_objectList.data.printStats();
		std::cerr<<"Created a new arrangement object. Key: "<<arrangementKey_objectList.arrangement_key<<std::endl;
		dataMap[part_nr].push_back(arrangementKey_objectList);
	}
}

int ObjectPartHash::buildModel(std::string data)
{
	//we can parse a string here in order to get parameters in
	std::cerr<<"Building model for OPH calssifier!"<<std::endl;
	std::cerr<<"Input Training dataset details: "<<std::endl;
	icf::DS::MatrixPtr extractedFeatures_ptr = this->trainDS->getFeatureMatrix("x");
	icf::DS::Matrix extractedFeatures= *extractedFeatures_ptr;
	icf::DS::MatrixPtr lMatrix_ptr =  this->trainDS->getFeatureMatrix("y");
	icf::DS::Matrix lMatrix = *lMatrix_ptr;
	icf::DS::MatrixPtr aMatrix_ptr = this->trainDS->getFeatureMatrix("arr");
	icf::DS::Matrix aMatrix = *aMatrix_ptr;
	icf::DS::MatrixPtr pMatrix_ptr =  this->trainDS->getFeatureMatrix("partNr");
	icf::DS::Matrix pMatrix = *pMatrix_ptr;


	std::cerr<<"Size of features in DS: Descriptors: "<<extractedFeatures.rows();
	std::cerr<<" labels: "<<lMatrix.rows()<<" arrangements: "<<aMatrix.rows()<<" partNr: "<<pMatrix.rows()<<std::endl;
	if ( (lMatrix.rows() == aMatrix.rows()) && (pMatrix.rows() == extractedFeatures.rows()) &&
			(extractedFeatures.rows() == aMatrix.rows()) && (pMatrix.rows() == lMatrix.rows()) )
	{
		for (int i = 0;i< lMatrix.rows(); ++i )
		{
			int part_nr = static_cast<int> (pMatrix(i,0));
			int arr_key = static_cast<int>(aMatrix(i,0));
			int ID = static_cast<int> (lMatrix(i,0));
			Eigen::VectorXf descriptor (extractedFeatures.cols());
			for(int j=0; j<extractedFeatures.cols();++j)
				descriptor(j) = static_cast<float>( extractedFeatures(i,j));
			addToHashTable(part_nr,arr_key ,ID,descriptor);
		}
	}

	std::cerr<<"OPH MODEL BUILT"<<std::endl;
	return 0;
}

//core classification method
std::map<int, std::vector<double> >  ObjectPartHash::vote(std::string input,  std::vector<int> &ground_truth_for_groups)
{
	std::cerr<<"VOTING!!!";
	std::map<int, std::vector<double> > partId_result;
	std::vector<std::set<int> > results_id_list;
	std::vector<std::vector< double> > result;//results for every grouping of segments

	//results based on part ID-ss
	int size_of_object=0;//number of points in an object
	std::vector<int> size_of_part;//number of points in a part

	std::vector<std::string> parsed_input;
	splitNonEmpty(parsed_input, input, "\n");
	std::cerr<<"NR OF LINES: "<<parsed_input.size()<<std::endl;
	std::vector<std::string> temp;//first line contains the number of parts an object is made out of as well as the sizes
	splitNonEmpty(temp, parsed_input[0], ", ()");
	int number_of_parts = atoi(temp.at(0).c_str());
	size_of_part.resize(number_of_parts);
	//read number of points in each part
	for(int j = 0;j<number_of_parts; ++j)
	{
		size_of_part[j] = atoi(temp.at(j+1).c_str());
		size_of_object += size_of_part[j];
	}
	std::cerr<<"NR of POINTS in OBJECT: "<<size_of_object<<std::endl;
	results_id_list.resize(parsed_input.size()-1);
	result.resize(parsed_input.size()-1);
	std::vector<int> ground_truth_for_parts(number_of_parts);
	ground_truth_for_groups.resize(parsed_input.size()-1);

	for(std::vector<vector<double> >::iterator it = result.begin(); it!= result.end();++it)
		it->resize(max_ID+1,0.0);

	for(unsigned int i = 1;i<parsed_input.size();++i)
	{
		int iterator=0;
		int size;//number of points in the segment
		std::vector<int> part_IDs; //list of partIDs segment is composed of

		std::vector<std::string> values;
		splitNonEmpty(values, parsed_input[i], ", ()");

		ground_truth_for_groups[i-1] = atoi(values.at(iterator++).c_str());//0
std::cerr<<"GR_TR: "<<ground_truth_for_groups[i-1]<<std::endl;
		int part_nr = atoi(values.at(iterator++).c_str());//1
std::cerr<<"PART_NR: "<<part_nr<<std::endl;
		//part_IDs.resize(part_nr);
		int arrangement_key = atoi(values.at(iterator++).c_str());//2
std::cerr<<"ARR:KEY:: "<<arrangement_key<<std::endl;
		for(int j = 0;j<part_nr;++j)
			part_IDs.push_back( atoi(values.at(iterator++).c_str()));
		size = atoi(values.at(iterator++).c_str());
std::cerr<<"GOT SIZE as WELL: "<<size<<std::endl;
		// if the group consists of a single part save it's ground truth
		if(part_IDs.size()==1)
			ground_truth_for_parts[part_IDs[0]] = ground_truth_for_groups[i-1];

		int grown_from = atoi(values.at(iterator++).c_str());
		int descr_length = atoi(values.at(iterator++).c_str());
std::cerr<<"Descr. Length: "<<descr_length<<std::endl;
		Eigen::VectorXf descriptor(descr_length);
		for(int j = 0;j<descr_length;j++)
			descriptor[j]=atof(values.at(iterator++).c_str());

		//build up a possible ID list of results for the current grouping
		//at the moment it always contains all the ID-s in a dataset
		std::set<int> id_list;
		if(grown_from == -1  || (grown_from != -1 && results_id_list[grown_from].empty()))
			for (int j=0; j<max_ID+1; ++j)
				id_list.insert(j);
		else
			id_list = results_id_list[grown_from];

		std::vector<int> nr_of_IDs_in_arrangement;
		//vector of pairs returned by the core classification method
		std::vector<std::pair<int, double> > i_values;

		std::vector<Arrangement>::iterator it_Arrangements = dataMap[part_nr].begin();
		for(;it_Arrangements < dataMap[part_nr].end();it_Arrangements++)
		{
			if(it_Arrangements->arrangement_key == arrangement_key)
			{
				i_values = it_Arrangements->search(id_list, descriptor,this->cl_type);
//				std::cerr<<"RETURNED: "<<i_values.size()<<std::endl;
				//TODO for debuging
				nr_of_IDs_in_arrangement = it_Arrangements->ID_counter;
				break;
			}
		}
		///////////////////////////METHOD SPECIFIC PART BEGINS///////////////////////////////////////////////
		std::set<int> IDs_in_result;
		for(std::vector<std::pair<int, double> >::iterator p_it = i_values.begin(); p_it!= i_values.end();++p_it)
			IDs_in_result.insert(p_it->first);
		results_id_list[i-1] = IDs_in_result;
		//interpret result (weighten result with partNr) for GROUPS
		for(std::vector<std::pair<int, double> >::iterator it = i_values.begin(); it!=i_values.end();++it)
			result[i-1][it->first] += (getResultWeight(part_nr) * it->second); //(double)part_nr/nr_of_IDs_in_arrangement[it->first];
		//store results based on partID
		for(unsigned j = 0; j<part_IDs.size();++j)
		{
			partId_result[ part_IDs[j] ].resize(max_ID+1,0.0);
			for(unsigned k=0; k<result[i-1].size(); ++k)
			{
				partId_result[ part_IDs[j] ].at(k) += (result[i-1][k]);//* (size_of_part[ part_IDs[j]]/(double) size );
			}
		}
	}
	return partId_result;
}


void ObjectPartHash::addData(std::string input)
{
//	double t1 = my_clock();
//	std::map<int, std::vector<double> > partId_result;//results based on part ID-ss
//	std::vector<int> ground_truth_for_groups;
//	result_string.str("");
//	std::cerr<<input<<std::endl;
//	std::cerr<<"Max ID is: "<<max_ID<<std::endl;
//
//	partId_result = vote(input, ground_truth_for_groups);
//
//	//print out results per partID
//	result_string<<max_ID<<std::endl;
//	for(std::map<int,std::vector<double> >::iterator it = partId_result.begin(); it!=partId_result.end(); ++it)
//	{
//		std::cerr<<"PART ID: "<<it->first<< "  Result: ";
//		// normalize(it->second);
//		for( std::vector<double>::iterator dit = it->second.begin(); dit!=it->second.end(); ++dit )
//		{
//			std::cerr<<*dit<<" ";
//			result_string<<*dit<<" ";
//		}
//		std::cerr<<std::endl;
//		result_string<<std::endl;
//	}
//	double t2 =my_clock();
//	ROS_ERROR("Time it took to classify: %f",t2-t1);
}

//void ObjectPartHash::addEvaluationData(std::string input)
//{
//
//  std::map<int, std::vector<double> > partId_result;
//  std::vector<int> ground_truth_for_groups;
//  double t1 = my_clock();
//  std::cerr<<"This is empty: "<<result_string.str()<<std::endl;
//
//  result_string.str("");
//
//  partId_result = classify(input, ground_truth_for_groups);
//
//  //calculating final result based on partID-sd
//  std::vector<double> result_based_on_partID(max_ID+1, 0.0);
//  for(std::map<int,std::vector<double> >::iterator it = partId_result.begin(); it!=partId_result.end(); ++it)
//    for( std::vector<double>::iterator dit = it->second.begin(); dit!=it->second.end(); ++dit )
//      result_based_on_partID[dit - it->second.begin()] += (*dit); //* exp(-( (double)size_of_object/size_of_part[it->first]));;
//  double max_score = -DBL_MAX;
//  int best_ID =-1;
//  double sum_of_elems =std::accumulate(result_based_on_partID.begin(),result_based_on_partID.end(),0.0);
//  // normalize(result_based_on_partID);
//  result_string << result_based_on_partID.size()<< " ";
//  //for (std::vector<double>::iterator it = final_result_counter.begin(); it!= final_result_counter.end(); ++it)
//  for (std::vector<double>::iterator it = result_based_on_partID.begin(); it!= result_based_on_partID.end(); ++it)
//  {
//    //*it = *it/sum_of_elems;
//    result_string<<*it<<" ";
//    //while writing it to string we find the best match in order to calculate conf Matrix
//    result_matrix(ground_truth_for_groups[0],it-result_based_on_partID.begin()) += *it/sum_of_elems;
//    if(*it >max_score)
//    {
//      max_score = *it;
//      //best_ID = it - final_result_counter.begin();
//      best_ID = it - result_based_on_partID.begin();
//    }
//  }
//  result_string<<std::endl;
//
//  updateTpr(result_based_on_partID, max_ID, ground_truth_for_groups[0]);
//
//  if(max_score>0)
//    confusionMatrix(ground_truth_for_groups[0],best_ID)++;
//
//  //the result is the list of scores for each part
//  for(std::map<int,std::vector<double> >::iterator it = partId_result.begin(); it!=partId_result.end(); ++it)
//  {
//    result_string<<it->first<<" ";
//    for( std::vector<double>::iterator dit = it->second.begin(); dit!=it->second.end(); ++dit )
//      result_string<<*dit<<" ";
//    result_string<<std::endl;
//  }
//  //std::cerr<<result_string.str();
//  double t2 =my_clock();
//  class_time +=(t2-t1);
//  cl_nr++;
//  ROS_WARN("Time it took to evaluate: %f",t2-t1);
//}
//
int ObjectPartHash::addTrainingExample(std::string input) //TODO parse based on parantheses first
{
	std::vector<std::string> values;
	splitNonEmpty(values,input, ", ()");
	int part_nr = atoi(values.at(0).c_str());
	int arrangement_key = atoi(values.at(1).c_str());
	int ID= atoi(values.at(2).c_str());
	Eigen::VectorXf descriptor(values.size()-3);
	for(unsigned int j = 3;j!=values.size();j++)
		descriptor[j-3]=atof(values.at(j).c_str());
	addToHashTable(part_nr, arrangement_key,ID,descriptor);
	return ID;
}

void ObjectPartHash::addTrainingData(std::string input)
{
	std::vector<std::string> values;
	splitNonEmpty(values,input, "\n");
	std::vector<std::string>::iterator st_it = values.begin();
	int ID;
	for(;st_it!=values.end();++st_it){
		//cerr<<"Adding Training Example: "<<*st_it<<endl;
		ID = addTrainingExample(*st_it);
	}
}
//
//void ObjectPartHash::save(std::string filename)
//{
//  std::ofstream  outfile(filename.c_str());
//  if(!outfile)
//  {
//    cerr<<"Unable to open file "<<filename;
//  }
//  std::stringstream out;
//  map<int,std::vector<Arrangement> >::iterator dataMap_it = dataMap.begin();
//  //int max_ID=0;
//  for (; dataMap_it != dataMap.end(); ++dataMap_it)
//  {
//    std::vector<Arrangement>::iterator arr_it = dataMap_it->second.begin();
//
//    for(;arr_it != dataMap_it->second.end() ; ++arr_it)
//    {
////      std::stringstream ds_out_fn;
////      ds_out_fn<<filename<<"_"<<arr_it->arrangement_key<<".ds";
////      ofstream out_ds(ds_out_fn.str().c_str(),ios::binary | ios::trunc);
////      out_ds<<arr_it->data;
////      out_ds.close();
//      for(std::vector<ObjectPartExamples>::const_iterator i=arr_it->objects.begin(); i!=arr_it->objects.end();++i)
//      {
//        for(std::multimap <float, Eigen::VectorXf>::const_iterator mm_it = i->descriptorList.begin();
//            mm_it!=i->descriptorList.end();
//            ++mm_it)
//        {
//          out<<"( "<<dataMap_it->first<<","<<arr_it->arrangement_key<<","<<i->ID<<")";
//          out<<"( "<<mm_it->second.transpose()<<" )"<<endl;
//        }
//      }
//    }
//  }
//  outfile<<out.str();
//}
//
std::string ObjectPartHash::classify()
{
	std::cerr<<"CLASSIFYING!"<<std::endl;
	//this SUX!!!!!!!
	icf::DS::MatrixPtr extractedFeatures_ptr = this->classifyDS->getFeatureMatrix("x");//the feature descriptor
	icf::DS::Matrix extractedFeatures= *extractedFeatures_ptr;
	std::cerr<<"EXTRACTED FEATURE SIZE: "<<extractedFeatures.rows()<<std::endl;
	icf::DS::MatrixPtr label_ptr =  this->classifyDS->getFeatureMatrix("y");//the label
	icf::DS::Matrix label = *label_ptr;
	icf::DS::MatrixPtr arr_ptr = this->classifyDS->getFeatureMatrix("arr");//arrangement key
	icf::DS::Matrix arr = *arr_ptr;
	icf::DS::MatrixPtr part_nr_ptr =  this->classifyDS->getFeatureMatrix("part_nr");//nr of parts in the grouping
	icf::DS::Matrix part_nr = *part_nr_ptr;
	icf::DS::MatrixPtr size_of_parts_ptr =  this->classifyDS->getFeatureMatrix("size_of_parts");//size of parts in the grouping
	icf::DS::Matrix size_of_parts = *size_of_parts_ptr;
	icf::DS::MatrixPtr partIds_ptr = this->classifyDS->getFeatureMatrix("partIds");//Ids of parts in the grouping
	icf::DS::Matrix partIds = *partIds_ptr;
	icf::DS::MatrixPtr group_nr_of_points_ptr = this->classifyDS->getFeatureMatrix("group_size");//nr of points in this grouping
	icf::DS::Matrix group_nr_of_points = *group_nr_of_points_ptr;
	icf::DS::MatrixPtr seed_ptr = this->classifyDS->getFeatureMatrix("seed");//seed group or whatever
	icf::DS::Matrix seed = *seed_ptr;
	std::stringstream out;
	this->labelMap = new LabelMap(this->classifyDS->getY());
	out<<size_of_parts.rows()<<",";

	for (int i = 0;i< size_of_parts.rows(); ++i)
		out<<size_of_parts(i,0)<<",";
	out<<std::endl;

	for(int i=0;i<label.rows();++i)
	{
		out << "(" << label(i,0) << "," << part_nr(i,0) << ","<< arr(i,0) <<",";
		for (int j= 1; j<=partIds(i,0);++j)
			out << partIds(i,j) <<",";
		out<<group_nr_of_points (i,0) <<","<<seed(i,0) <<")";
		out <<"("<<extractedFeatures.row(i).cols()<<" "<<extractedFeatures.row(i)<<")";
		out <<std::endl;
	}
	std::cerr<<out.str()<<std::endl;
	std::map<int, std::vector<double> > partId_result;
	std::vector<int> ground_truth_for_groups;
	result_string.str("");
	partId_result = vote(out.str(), ground_truth_for_groups);
	result_string<<max_ID<<std::endl;
	std::cerr<<"AFTER VOTE!!"<<std::endl;


	ClassificationResult result(*this->labelMap);
	DataSet<double>::MatrixPtr confsPtr;
	confsPtr = DataSet<double>::MatrixPtr((new DataSet<double>::Matrix(partId_result.size(),this->max_ID+1	)));
	//result->confidences->conservativeResize(partId_result.size(),this->max_ID);
	std::cerr<<"CONFIDENCES SIZE: "<<(*confsPtr).rows()<<" x "<< (*confsPtr).cols()<<std::endl;
	for(std::map<int,std::vector<double> >::iterator it = partId_result.begin(); it!=partId_result.end(); ++it)
	{
		// normalize(it->second);
		int prediction;
		std::vector<double>::iterator max_it = std::max_element(it->second.begin(), it->second.end());
		prediction = max_it - it->second.begin();
		result.add(prediction);
		for(int i=0;i<it->second.size();++i){
			(*(confsPtr))(it->first,i) =static_cast<double>( it->second[i]);
		}
		for( std::vector<double>::iterator dit = it->second.begin(); dit!=it->second.end(); ++dit )
		{
			result_string<<*dit<<" ";
		}
		result_string<<std::endl;
	}
	std::cerr<<"RESULT STRING: "<<std::endl<<result_string.str()<<std::endl;
	std::stringstream ss;

	std::cerr<<"LABEL MAP: "<<*this->labelMap<<std::endl;
	result.confidences = confsPtr;
	std::cerr<<"RESULT ON SERVER SIDE: "<<std::endl<<*result.confidences<<std::endl;
	ss<<result;
	return 	ss.str();
}
void ObjectPartHash::load(std::string filename)
{
	std::cerr<<"[note] Loading data from a saved file..."<<std::endl;

	std::ifstream infile(filename.c_str());
	std::string line;
	std::stringstream stream;
	if (infile.is_open())
	{
		std::cerr<<"[note] succesfully opened file "<<std::endl;
		while ( infile.good() )
		{
			getline (infile,line);
			stream << line << endl;
		}
		infile.close();
		std::cerr<<"[note] succesfully read file to stream...adding as training data"<<std::endl;
		addTrainingData(stream.str());
	}

	else std::cerr << "Unable to open file";
}

void ObjectPartHash::printData()
{
  //debugging purposes
  map<int,std::vector<Arrangement> >::iterator it = dataMap.begin();
  //int max_ID=0;
  for (; it != dataMap.end(); ++it)
  {
    std::vector<Arrangement>::iterator v_it = it->second.begin();
    for(;v_it != it->second.end() ; ++v_it)
    {
      cerr<<"    part_Nr:  "<<it->first<<endl;
      cerr<<"        Key:  "<<v_it->arrangement_key<<endl;
      cerr<<"IDs with this arrangement Key:"<<endl;
      //     v_it->
      for(std::vector<int>::iterator v_i = v_it->ID_counter.begin(); v_i!=v_it->ID_counter.end();++v_i)
      {
        cerr<<"           "<<v_i-v_it->ID_counter.begin()<< " : "<<*v_i<<std::endl;
      }
      cerr<<"---------";
      for(std::vector<ObjectPartExamples>::const_iterator i=v_it->objects.begin(); i!=v_it->objects.end();++i)
      {
        /*        if(i->ID > max_ID)
          max_ID=i->ID;
        cerr<<"         ID:  "<<i->ID<<endl;*/

        for(std::multimap <float, Eigen::VectorXf>::const_iterator mm_it = i->descriptorList.begin();
            mm_it!=i->descriptorList.end();
            ++mm_it)
        {
          cerr<<"Description: "<<mm_it->second.transpose()<<endl;
          cerr<<"=============="<<endl;
        }
        cerr<<"---------"<<endl;
      }
      cerr<<"-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"<<endl;
    }
  }
  //return max_ID;
}
//
//void ObjectPartHash::updateTpr(const std::vector<double> result, const int max_ID, const int ground_truth)
//{
//  int k=0;
//  for(k=0;k<=max_ID+1;++k)
//  {
//    std::vector<int> search_res = nMax(result, k);
//
//    std::vector<int>::iterator found =  std::find(search_res.begin(),search_res.end(), ground_truth);
//
//    if(found != search_res.end())
//      tpr(k) +=1;
//  }
//}
//
//std::vector<int> ObjectPartHash::nMax (std::vector<double>  vect, int n)
//{
//  std::vector<int> result;
//  for(int i=0;i<n;++i)
//  {
//    std::vector<double>::iterator max = std::max_element(vect.begin(),vect.end());
//    result.push_back(max - vect.begin());
//    *max = 0;
//  }
//  return result;
//}
//
//int ObjectPartHash::buildModel (std::string st){
//
//  if(built){
//    ROS_INFO("MODEL ALLREADY BUILT");
//    return 1;
//  }
//  else if((cl_type == c_NN) && !built)
//  {
//    ROS_INFO ("NO BUILD NEEDED for NN");
//    built=true;
//    return 1;
//  }
//  else if( (cl_type == c_kNN) && !built)//need !built variable so when I test separate files and run the bash script it only gets built once
//  {
//    ROS_INFO ("BUILDING MODEL FOR kNN");
//    for( std::map<int,std::vector<Arrangement> >::iterator it = dataMap.begin();it!=dataMap.end();++it)
//      for(std::vector<Arrangement>::iterator arr_i = it->second.begin(); arr_i!=it->second.end(); ++arr_i)
//      {
//
//        std::cerr<<"ARRANGEMENT: "<<arr_i->arrangement_key<<std::endl;
//        std::ostringstream temp_oss;
//        temp_oss<<arr_i->data;
//        arr_i->classifier = new icf::KNNClassifier<double>("-k 3");
//        arr_i->classifier->addTrainingData(temp_oss.str());
//        arr_i->classifier->setDataset(arr_i->data,"train");
//        arr_i->classifier->buildModel("");
//        std::cerr<<"MODEL BUILT"<<std::endl;
//      }
//    built=true;
//    return 1;
//  }
//  else if((cl_type == c_SVM) && !built)
//  {
//    ROS_INFO ("BUILDING MODEL FOR SVM");
//    for( std::map<int,std::vector<Arrangement> >::iterator it = dataMap.begin();it!=dataMap.end();++it)
//         for(std::vector<Arrangement>::iterator arr_i = it->second.begin(); arr_i!=it->second.end(); ++arr_i)
//         {
//           std::cerr<<"ARRANGEMENT: "<<arr_i->arrangement_key<<std::endl;
//           std::ostringstream temp_oss;
//           temp_oss<<arr_i->data;
//           arr_i->classifier = new icf::SVMClassifier("");
//         }
//    built=true;
//    return 1;
//  }
//  return -1;
//
//}

//void ObjectPartHash::addTrainingData(std::string data)
//{
//	this->trainingData = data;
//}
//
//void ObjectPartHash::addData(std::string data)
//{
//	this->classificationData = data;
//}
//
//void ObjectPartHash::addEvaluationData(std::string evaluationData)
//{
//	this->evaluationData = evaluationData;
//}
