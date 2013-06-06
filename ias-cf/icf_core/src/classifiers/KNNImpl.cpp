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

#include <icf_core/classifiers/KNNImpl.h>
namespace icf
{
using namespace boost;
using namespace boost::lambda;
using namespace std;

//#define DEBUGOUT

template<class T>
  vector<int> maxKeys(const std::map<int, T>& map)
  {
    vector<int> maxKeys;
    if (map.size() == 0)
      return maxKeys;
    maxKeys.push_back(map.begin()->first);
    T maxVal = map.begin()->second;

    for (typename std::map<int, T>::const_iterator iter = map.begin(); iter != map.end(); iter++)
    {
      if (iter->second > maxVal)
      {
        maxKeys.clear();
        maxKeys.push_back(iter->first);
        maxVal = iter->second;
      }
      else if (iter->second == maxVal)
      {
        maxKeys.push_back(iter->first);
      }
    }
    return maxKeys;
  }

/**
 * Sample a position between start and end. Start and end can be iterators
 */
template<class T>
  T sample_uniform(T start, T end)
  {
    int range = end - start;
    int i = rand() % range;
    return start + i;
  }

template<class T>
  void printMap(const char * name, const std::map<int, T>& map)
  {
#ifdef DEBUGOUT
    std::cout<<std::endl<<name<<std::endl;
    typename std::map<int,T>::const_iterator start=map.begin();
    typename std::map<int,T>::const_iterator end = map.end();
    while(start!=end)
    {
      std::cout<<"("<<start->first<<","<<start->second<<")"<<" ";
      start++;
    }
    std::cout<<std::endl;
#endif
  }

template<class T>
  void printIter(const char* name, T start, T end)
  {
#ifdef DEBUGOUT
    std::cout<<std::endl<<name<<std::endl;
    while(start!=end)
    {

      std::cout<<*(start)<<" ";;
      start++;
    }
    std::cout<<std::endl;
#endif
  }

template<class T>
  void printMatrix(const char * name, const flann::Matrix<T>& matrix)
  {
#ifdef DEBUGOUT
    std::cout<<std::endl<<name<<std::endl;
    for(unsigned int i=0;i<matrix.rows;i++)
    {
      for(unsigned int j=0;j<matrix.cols;j++)
      {
        std::cout<<matrix[i][j]<<" ";
      }
      std::cout<<std::endl;
    }
    std::cout<<endl;
#endif
  }

template<class Metric>
  KNNImpl<Metric>::KNNImpl(unsigned int k, shared_ptr<flann::Index<Metric> > index, shared_ptr<vector<int> > labels,
                           bool weighVotes) :
      k(k), index(index), labels(labels), weighVotes(weighVotes)
  {
    if (k > labels->size())
    {
      throw ICFException() << invalid_argument("k greater than number of instances");
    }

  }

template<class Metric>
  KNNImpl<Metric>::KNNImpl(const flann::Matrix<typename Metric::ElementType>& dataset, std::string baseName,
                           unsigned int k, bool weighVotes, Metric m) :
      k(k), weighVotes(weighVotes)
  {
	std::cerr<<"KNNIMPL:: CONSTRUCTOR BEGIN!!"<<std::endl;
    std::string labelsFile = baseName + std::string(".labels.knn");
    std::string indexFile = baseName + std::string(".index.knn");
    std::cerr<<"KNNIMPL::FILES:"<<std::endl;
    std::cerr<<labelsFile<<std::endl<<indexFile<<std::endl;

    std::ifstream lf(labelsFile.c_str(), std::ios_base::in);
    if (lf.fail())
    {
      std::cerr<<"KNNIMPL::LABELS READ IN FAILED!"<<std::endl;
    	throw ICFException() << invalid_state_error("Can't open label file for reading");
    }
    this->labels = boost::shared_ptr<std::vector<int> >(new std::vector<int>());
    int i;
    while (!lf.eof())
    {
      lf >> i;
      this->labels->push_back(i);
    }

    //TODO: Can't use minkowski distance with this
    std::cerr<<"KNNIMPL::LABELS WERE READ IN!"<<std::endl;
    this->index = boost::shared_ptr<flann::Index<Metric> >(new flann::Index<Metric>(dataset, flann::SavedIndexParams(indexFile), m));
    std::cerr<<"KNNIMPL::INDEX FILE READ IN"<<std::endl;
    printMatrix<typename Metric::ElementType>("Dataset loaded:", dataset);
    std::cerr<<"IF HERE SUCCES!"<<std::endl;
  }

template<class Metric>
  KNNImpl<Metric>::~KNNImpl()
  {

  }

template<class Metric>
  void KNNImpl<Metric>::save(std::string baseName)
  {
    std::string labelsFile = baseName + std::string(".labels.knn");
    std::string indexFile = baseName + std::string(".index.knn");

    std::fstream lf;
    lf.open(labelsFile.c_str(), std::ios_base::out);
    if (lf.fail())
    {
      throw ICFException() << invalid_state_error("Can't open label file for writing");
    }

    for (unsigned int i = 0; i < this->labels->size(); i++)
    {
      lf << labels->at(i) << " ";
    }
    lf << flush;
    lf.close();
    this->index->save(indexFile);
  }

/**
 * Tries to classify instances by conducting a k-nearest neighbour search and
 * voting. Votes are weighed by 1/(dist^2) if weighing option is set. Ties are broken by
 * uniformly sampling one of the labels.
 * @return a tuple containing a shared_ptr to a vector containg the predictions
 * a matrix containing the nearest neighbours and a matrix containing the distances to
 * the nearest neighbours
 */
template<class Metric>
  typename KNNImpl<Metric>::KNNResult KNNImpl<Metric>::classify(
      flann::Matrix<typename Metric::ElementType>& instances) const
  {

    printMatrix("Instances", instances);

    srand(time(NULL));
    vector<int>& labels = *(this->labels.get());

    printIter("Labels", labels.begin(), labels.end());

    shared_ptr<vector<int> > predictions(new vector<int>(instances.rows));

    flann::Matrix<int>& indices = *new flann::Matrix<int>(new int[instances.rows * k], instances.rows, k);

    typedef typename Metric::ElementType ET;
    flann::Matrix<typename Metric::ResultType>& dists = *new flann::Matrix<typename Metric::ResultType>(
        new ET[instances.rows * k], instances.rows, k);

    index->knnSearch(instances, indices, dists, k, flann::SearchParams(-1));

    printMatrix("nn-indices: ", indices);
    printMatrix("nn-distances:   ", dists);

    ET * nearestDists = NULL;
    int * nearestIndices = NULL;
    int index = 0;
    std::map<int, int> votes;
    std::map<int, double> wVotes;
    for (size_t i = 0; i < instances.rows; i++)
    {
      nearestDists = dists[i];
      nearestIndices = indices[i];
      index = 0;
      votes.clear();
      wVotes.clear();
      if (weighVotes)
      {
#ifdef DEBUG
        cout<<"Weighing votes"<<endl;
#endif
        //Exact matches handling

        //Vote, but count only exact matches
        for_each(nearestDists, nearestDists + k,
                 (if_then(_1 == 0.0, var(votes)[var(labels)[var(nearestIndices)[var(index)]]]++), var(index)++));

        //if there are any exact matches....
        if (votes.size() != 0)
        {
          vector<int> mi = maxKeys<int>(votes);

          if (mi.size() <= labels.size())
          {
            predictions->at(i) = *sample_uniform<vector<int>::iterator>(mi.begin(), mi.end());
            continue;
          }
        }
        //No exact matches -> conduct weighing
        index = 0;
        //Vote
        for_each(nearestDists, nearestDists + k,
                 (var(wVotes)[var(labels)[var(nearestIndices)[var(index)]]] += (1 / (_1 * _1)), var(index)++));
        printMap("Weighed votes", wVotes);
        //predict
        vector<int> mi = maxKeys<double>(wVotes);
        predictions->at(i) = *sample_uniform<vector<int>::iterator>(mi.begin(), mi.end());

      }
      else
      {
#ifdef DEBUG
        cout<<"Not weighing votes"<<endl;
#endif
        for_each(nearestIndices, nearestIndices + k, var(votes)[var(labels)[_1]]++);
        vector<int> mi = maxKeys<int>(votes);
        predictions->at(i) = *sample_uniform<vector<int>::iterator>(mi.begin(), mi.end());

      }

    }
    shared_ptr<flann::Matrix<int> > indicesPtr(&indices);
    shared_ptr<flann::Matrix<typename Metric::ResultType> > distPtr(&dists);
    printIter("Predictions ", predictions->begin(), predictions->end());
    return make_tuple<shared_ptr<vector<int> >, shared_ptr<flann::Matrix<int> >,
        shared_ptr<flann::Matrix<typename Metric::ResultType> > >(predictions, indicesPtr, distPtr);
  }

template class KNNImpl<flann::L2<double> > ;
template class KNNImpl<flann::L2<float> > ;
template class KNNImpl<flann::L1<double> > ;
template class KNNImpl<flann::L1<float> > ;
template class KNNImpl<flann::MinkowskiDistance<double> > ;
template class KNNImpl<flann::HellingerDistance<double> > ;
template class KNNImpl<flann::KL_Divergence<double> > ;
template class KNNImpl<flann::HistIntersectionDistance<double> > ;
template class KNNImpl<flann::ChiSquareDistance<double> > ;
template class KNNImpl<flann::MinkowskiDistance<float> > ;
template class KNNImpl<flann::HellingerDistance<float> > ;
template class KNNImpl<flann::KL_Divergence<float> > ;
template class KNNImpl<flann::HistIntersectionDistance<float> > ;
template class KNNImpl<flann::ChiSquareDistance<float> > ;

}
