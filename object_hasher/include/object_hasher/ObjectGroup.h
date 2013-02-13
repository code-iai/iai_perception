/*
 * ObjectComponent.h
 *
 *  Created on: Mar 15, 2011
 *      Author: ferka
 */
#include <Eigen/Core>
#include <vector>



#ifndef OBJECTGROUP_H_
#define OBJECTGROUP_H_

using namespace std;

class ObjectGroup
{
  public:
    Eigen::VectorXi group_;
    Eigen::VectorXf descriptor_;
    Eigen::VectorXi partIDs;

    ObjectGroup():grown_form(0),ID_(0),arrangement_key_(0),part_nr_(0),size(0)   {
    }
    int CalculateArrangementKey(std::vector<int> grade_list){
    	 int pow = 1;
    	  //int key=0;
    	  part_nr_ = grade_list.size();
    	  for (std::vector<int>::iterator it = grade_list.begin(); it != grade_list.end(); ++it)
    	  {
    	    //int temp = *it;
    	    arrangement_key_ += (pow * (abs(*it)));
    	    pow *= 10;
    	  }
    	  return arrangement_key_;
    }
    int CalculateArrangementKey(std::vector<int> grade_list, std::vector<int> eigenv){
    	 int pow = 1;
    	  //int key=0;
    	  part_nr_ = grade_list.size();
    	  for (std::vector<int>::iterator it = grade_list.begin(); it != grade_list.end(); ++it)
    	  {
    	    //int temp = *it;
    	    arrangement_key_ += (pow * (*it));
    	    pow *= 10;
    	  }
    	  for (std::vector<int>::iterator it = eigenv.begin(); it != eigenv.end(); ++it)
    	  {
    	    //int temp = *it;
    	    arrangement_key_ += (pow * (abs(*it)));
    	    pow *= 10;
    	  }

    	  return arrangement_key_;
    }
    void determinePartIdsList(){
      partIDs.resize(part_nr_);
      int j=0;
      for(int i = 0; i<group_.rows(); ++i)
        if(group_(i)==1)
          partIDs(j++) = i;
    }

    int grown_form;
    int ID_;
    int arrangement_key_;
    int part_nr_;
    int size;
};

#endif /* OBJECTGROUP_H_ */
