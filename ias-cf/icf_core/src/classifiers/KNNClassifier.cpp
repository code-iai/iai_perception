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

#include <icf_core/classifiers/KNNClassifier.h>
namespace icf
{

using namespace TCLAP;
using namespace boost;

//#define DEBUGOUT

template<class T>
  KNNClassifier<T>::KNNClassifier(std::string params) :
      classifier(NULL), data(NULL), classLabels(new std::vector<int>()), rebuildModel(true)
  {
    evaluationData = "";
    classificationData = "";
    parse_parameters(params);
    std::stringstream ss;
    ss << rand();
    ds_prefix = ss.str();

  }

template<class T>
  KNNClassifier<T>::~KNNClassifier()
  {
    if (this->classifier != NULL)
    {
      delete this->classifier;
    }
    if (this->data != NULL)
    {
      delete[] this->data->ptr();
      delete this->data;
    }
  }

template<class T>
  void printMatrix(const char * name, flann::Matrix<T>& matrix)
  {
#ifdef DEBUGOUT
    std::cout << std::endl << name << std::endl;
    for (unsigned int i = 0; i < matrix.rows; i++)
    {
      for (unsigned int j = 0; j < matrix.cols; j++)
      {
        std::cout << matrix[i][j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
#endif
  }

template<class T>
  void printIter(const char* name, T start, T end)
  {
#ifdef DEBUGOUT
    std::cout << std::endl << name << std::endl;
    while (start != end)
    {
      std::cout << *(start) << " ";
      ;
      start++;
    }
    std::cout << std::endl;
#endif
  }

template<class T>
  void KNNClassifier<T>::preprocessTrainingData(DS& ds)
  {
    DS::MatrixPtr x_ptr = ds.getX();
    DS::MatrixPtr y_ptr = ds.getY();
    DS::Matrix& x = *x_ptr;
    DS::Matrix& y = *y_ptr;

    labelMap = LabelMap(ds.getY());
    unsigned int cols = x.cols();

    int rowOffset = 0;
    if (this->data != NULL)
    {

      if (this->data->cols != cols)
      {
        throw ICFException() << invalid_state_error("Previously submitted data has a different number of columns");
      }

      rowOffset = this->data->rows;
      size_t newRows = rowOffset + y.rows();
      T * data = new T[newRows * cols];
      flann::Matrix<T> *newData = new flann::Matrix<T>(data, newRows, cols);
      memcpy(newData->ptr(), this->data->ptr(), sizeof(T) * this->data->rows * this->data->cols);
      delete[] this->data->ptr();
      delete this->data;
      delete this->classifier;
      this->classifier = NULL;
      this->data = newData;
    }
    else
    {
      T* data = new T[x.rows() * cols];
      this->data = new flann::Matrix<T>(data, x.rows(), cols);
    }

    for (int r = 0; r < x.rows(); r++)
    {
      for (int c = 0; c < x.cols(); c++)
      {
        this->data->operator []((rowOffset + r))[c] = x(r, c);
      }
    }
    for (int r = 0; r < y.rows(); r++)
    {
      this->classLabels->push_back(y(r, 0));
    }

    printMatrix("Training data", *(this->data));
    printIter("Training labels", this->classLabels->begin(), this->classLabels->end());
    rebuildModel = true;
  }

template<class T>
  void KNNClassifier<T>::addTrainingData(std::string data)
  {
    DS::Ptr ds_ptr(new DS("/" + ds_prefix + "_train", (void*)data.c_str(), data.length() * sizeof(char), false));
    preprocessTrainingData(*ds);
  }

/**
 *\brief
 */
template<class T>
  int KNNClassifier<T>::buildModel(std::string params)
  {

    if (params != "")
      parse_parameters(params);
    preprocessTrainingData(*this->trainDS);
    startTiming();
    //std::cout << "Building model" << std::endl;
    if (this->classifier != NULL)
      delete this->classifier;
    if (this->m == "L1")
    {
      boost::shared_ptr<flann::Index<flann::L1<T> > > index(
          new flann::Index<flann::L1<T> >(*(this->data), flann::KMeansIndexParams(32, 10)));
      index->buildIndex();
      this->classifier = new KNNImpl<flann::L1<T> >(this->k, index, this->classLabels, this->w);
    }
    else if (this->m == "L2")
    {
      boost::shared_ptr<flann::Index<flann::L2<T> > > index(
          new flann::Index<flann::L2<T> >(*(this->data), flann::KMeansIndexParams(32, 10)));
      index->buildIndex();
      this->classifier = new KNNImpl<flann::L2<T> >(this->k, index, this->classLabels, this->w);
    }
    else if (this->m == "minkowski")
    {
      boost::shared_ptr<flann::Index<flann::MinkowskiDistance<T> > > index(
          new flann::Index<flann::MinkowskiDistance<T> >(*(this->data), flann::KMeansIndexParams(32, 10),
                                                         flann::MinkowskiDistance<T>(this->mParam)));
      index->buildIndex();
      this->classifier = new KNNImpl<flann::MinkowskiDistance<T> >(this->k, index, this->classLabels, this->w);
    }
    else if (this->m == "histintersection")
    {
      boost::shared_ptr<flann::Index<flann::HistIntersectionDistance<T> > > index(
          new flann::Index<flann::HistIntersectionDistance<T> >(*(this->data), flann::KMeansIndexParams(32, 10)));
      index->buildIndex();
      this->classifier = new KNNImpl<flann::HistIntersectionDistance<T> >(this->k, index, this->classLabels);
    }
    else if (this->m == "hellinger")
    {
      boost::shared_ptr<flann::Index<flann::HellingerDistance<T> > > index(
          new flann::Index<flann::HellingerDistance<T> >(*(this->data), flann::KMeansIndexParams(32, 10)));
      index->buildIndex();
      this->classifier = new KNNImpl<flann::HellingerDistance<T> >(this->k, index, this->classLabels, this->w);
    }
    else if (this->m == "chisquared")
    {
      boost::shared_ptr<flann::Index<flann::ChiSquareDistance<T> > > index(
          new flann::Index<flann::ChiSquareDistance<T> >(*(this->data), flann::KMeansIndexParams(32, 10)));
      index->buildIndex();
      this->classifier = new KNNImpl<flann::ChiSquareDistance<T> >(this->k, index, this->classLabels, this->w);
    }
    else if (this->m == "kl")
    {
      boost::shared_ptr<flann::Index<flann::KL_Divergence<T> > > index(
          new flann::Index<flann::KL_Divergence<T> >(*(this->data), flann::KMeansIndexParams(32, 10)));
      index->buildIndex();
      this->classifier = new KNNImpl<flann::KL_Divergence<T> >(this->k, index, this->classLabels, this->w);
    }
    stopTiming();
    return 0;
  }

template<class T>
  void KNNClassifier<T>::save(std::string filename)
  {
    std::string datasetFile = filename + ".ds.knn";
    std::fstream datasetStream(datasetFile.c_str(), std::ios_base::out | std::ios_base::binary);
    if (!labelMap.save(filename + ".cls"))
    {
      throw ICFException() << invalid_state_error("Can't save label map to file " + filename + ".cls");
    }
    if (!datasetStream || datasetStream.bad())
    {
      throw ICFException() << invalid_state_error("Can't open file " + filename + ".ds.knn " + " for writing");
    }
    datasetStream.write((char*)&this->data->rows, sizeof(this->data->rows) / sizeof(char));
    datasetStream.write((char*)&this->data->cols, sizeof(this->data->cols) / sizeof(char));
    datasetStream.write((char*)&this->data->stride, sizeof(this->data->stride) / sizeof(char));
    datasetStream.write((char*)this->data->ptr(), this->data->rows * this->data->stride * sizeof(T) / sizeof(char));
    datasetStream.close();
    this->classifier->save(filename);
    if (this->cm)
      this->cm->save(filename + ".cm");

  }

template<class T>
  void KNNClassifier<T>::load(std::string filename)
  {
    std::cerr<<"FILE PARAM RECEIVED:"<<std::endl;
	std::string datasetFile = filename + std::string(".ds.knn");
    std::cerr<<"DATASET FILE: "<<datasetFile<<std::endl;
    std::ifstream datasetStream(datasetFile.c_str(), std::ios_base::binary);
    if (!labelMap.load(filename + ".cls"))
    {
      throw ICFException() << invalid_state_error("Can't load label map");
    }
    if (datasetStream)
    {
      std::cerr<<"SOME IF IN KNN LOAD"<<std::endl;
      size_t rows = 0;
      size_t cols = 0;
      size_t stride = 0;
      std::cerr<<"KNN::READING IN"<<std::endl;
      datasetStream.read((char*)&rows, sizeof(rows) / sizeof(char));
      if (datasetStream.eof())
        throw ICFException() << invalid_state_error("Invalid data file");
      datasetStream.read((char*)&cols, sizeof(cols) / sizeof(char));
      if (datasetStream.eof())
        throw ICFException() << invalid_state_error("Invalid data file");
      datasetStream.read((char*)&stride, sizeof(stride) / sizeof(char));
      if (datasetStream.eof())
        throw ICFException() << invalid_state_error("Invalid data file");
      T * data = new T[rows * stride];
      datasetStream.read((char*)data, rows * cols * sizeof(T) / sizeof(char));
      this->data = new flann::Matrix<T>(data, rows, cols, stride);
      std::cerr<<"KNN::THAT IF JUST ENDED"<<std::endl;
      std::cerr<<"COLS: "<<this->data->cols <<"ROWS: "<<this->data->rows<<std::endl;
    }

    else
    {
      throw ICFException() << invalid_state_error("Can't open data file for reading");
    }

    if (this->classifier != NULL)
      delete this->classifier;

    if (this->m == "L1")
    {
      this->classifier = new KNNImpl<flann::L1<T> >(*(this->data), filename, this->k, this->w);
    }
    else if (this->m == "L2")
    {
    	std::cerr<<"KNN::DISTANCE IS L2-> CREATING KNNimpl"<<std::endl;
    	this->classifier = new KNNImpl<flann::L2<T> >(*(this->data), filename, this->k, this->w);
    	std::cerr<<"KNN::KNNImpl CREATED! YAAAY"<<std::endl;
    }
    else if (this->m == "minkowski")
    {
      this->classifier = new KNNImpl<flann::MinkowskiDistance<T> >(*(this->data), filename, this->k, this->w,
                                                                   flann::MinkowskiDistance<T>(this->mParam));
    }
    else if (this->m == "histintersection")
    {
      this->classifier = new KNNImpl<flann::HistIntersectionDistance<T> >(*(this->data), filename, this->k, this->w);
    }
    else if (this->m == "hellinger")
    {
      this->classifier = new KNNImpl<flann::HellingerDistance<T> >(*(this->data), filename, this->k, this->w);
    }
    else if (this->m == "chisquared")
    {
      this->classifier = new KNNImpl<flann::ChiSquareDistance<T> >(*(this->data), filename, this->k, this->w);
    }
    else if (this->m == "kl")
    {
      this->classifier = new KNNImpl<flann::KL_Divergence<T> >(*(this->data), filename, this->k, this->w);
    }
    else
    {
      throw ICFException() << invalid_argument("Invalid metric");
    }

    std::cerr<<"KNN::GETING LABELS"<<std::endl;
    this->classLabels = this->classifier->getLabels();
    std::cerr<<"KNN::GOT LABELS...YAY"<<std::endl;

    std::cerr<<"KNN::GETING CM"<<std::endl;
    std::fstream in((filename + ".cm").c_str(), std::ios_base::in);
    if (in.good())
    {
      in.close();
      this->cm = shared_ptr<ConfusionMatrix>(new ConfusionMatrix(filename + ".cm"));
      std::cerr<<"KNN::GOT CM...YAY"<<std::endl;
    }
    std::cerr<<"KNN::THE END OF LOAD"<<std::endl;

  }

template<class T>
  void KNNClassifier<T>::addData(std::string data)
  {
    this->classificationData = data;
  }

template<class T>
  void KNNClassifier<T>::addEvaluationData(std::string evaluationData)
  {
    this->evaluationData = evaluationData;
  }

template<class T>
  std::string KNNClassifier<T>::evaluate()
  {
    startTiming();
    if (this->classifier == NULL)
    {
      throw ICFException() << invalid_state_error("Model has to be build first");
    }

    DS::Ptr ds;
    if (evaluationData != "")
    {
      ds.reset(
          new DataSet<double>("/" + ds_prefix + "_train", (void*)evaluationData.c_str(),
                              evaluationData.length() * sizeof(char), false));
    }
    else
    {
      ds = this->evalDS;
    }

    DS::MatrixPtr x_ptr = ds->getX();
    DS::Matrix& x = *x_ptr;
    DS::MatrixPtr y_ptr = ds->getY();
    DS::Matrix& y = *y_ptr;

    unsigned int cols = x.cols();

    boost::shared_ptr<std::vector<int> > classLabels(new std::vector<int>());
    T * dt = new T[x.rows() * cols];
    flann::Matrix<T> query(dt, x.rows(), cols);

    for (int r = 0; r < x.rows(); r++)
    {
      for (int c = 0; c < x.cols(); c++)
      {
        query[r][c] = x(r, c);
      }
    }

    for (int r = 0; r < y.rows(); r++)
    {
      classLabels->push_back(y(r, 0));
    }

    printMatrix("Test data", query);
    printIter("Test labels", classLabels->begin(), classLabels->end());

    typename IKNNImpl<T>::KNNResult result = classifier->classify(query);
    boost::shared_ptr<std::vector<int> > predictedLabels = result.get<0>();
    delete[] query.ptr();
    EvaluationResult evaluationResult(classLabels, predictedLabels);
    evaluationResult.confidences.reset(createDummyConfidenceMatrix(labelMap, predictedLabels));
    this->cm = evaluationResult.getConfusionMatrix();
    std::stringstream ss;
    ss << evaluationResult;
    delete[] result.get<1>()->ptr();
    delete[] result.get<2>()->ptr();
    stopTiming();
    //std::cout << ss.str() << std::endl;
    return ss.str();

  }

/**
 *\brief classifying/adding test data
 *\return result for given test data
 *\param data to be tested
 */
template<class T>
  std::string KNNClassifier<T>::classify()
  {
    startTiming();
    if (this->classifier == NULL)
    {
      throw std::string("You have to build the model first");
    }

    DS::Ptr ds;
    if (this->classificationData != "")
    {
      ds = shared_ptr<DS>(
          new DataSet<double>("/" + ds_prefix + "_train", (void*)classificationData.c_str(),
                              classificationData.length() * sizeof(char), false));
    }
    else
    {
      ds = this->classifyDS;
    }

    DS::MatrixPtr x_ptr = ds->getX();
    DS::Matrix& x = *x_ptr;

    unsigned int cols = x.cols();

    T * dt = new T[x.rows() * cols];
    flann::Matrix<T> query(dt, x.rows(), cols);

    for (int r = 0; r < x.rows(); r++)
    {
      for (int c = 0; c < x.cols(); c++)
      {
        query[r][c] = x(r, c);
      }
    }

    printMatrix("Test data", query);
    printIter("Test labels", classLabels->begin(), classLabels->end());

    typename IKNNImpl<T>::KNNResult result = classifier->classify(query);
    boost::shared_ptr<std::vector<int> > predictedLabels = result.get<0>();
    delete[] query.ptr();
    ClassificationResult cr(labelMap, predictedLabels);
    cr.confidences.reset(createDummyConfidenceMatrix(labelMap, predictedLabels));
    std::stringstream ss;
    ss << cr;
    delete[] result.get<1>()->ptr();
    delete[] result.get<2>()->ptr();
    stopTiming();
    return ss.str();
  }

template<class T>
  std::string KNNClassifier<T>::executeService(std::string service, std::string params)
  {
    if (service == "clear_training_data")
    {
      if (this->data != NULL)
      {
        delete[] this->data->ptr();
        delete this->data;
        this->data = NULL;
        this->classLabels->clear();
      }
      return "cleared";
    }
    else
    {
      return Classifier::executeService(service, params);
    }
  }

template<class T>
  bool KNNClassifier<T>::parse_parameters(std::string params)
  {
    TCLAP::CmdLine cmdLine(" ", ' ', " ", false);
    ValueArg<int> kArg("k", "k", "the k part of knn", false, 1, "int");
    std::vector<std::string> availableMetrics;
    availableMetrics.push_back("L1");
    availableMetrics.push_back("L2");
    availableMetrics.push_back("minkowski");
    //availableMetrics.push_back("max");
    availableMetrics.push_back("histintersection");
    availableMetrics.push_back("hellinger");
    availableMetrics.push_back("chisquared");
    availableMetrics.push_back("kl");
    ValuesConstraint<std::string> availableMetricsConstraints(availableMetrics);
    ValueArg<std::string> metricArg("m", "metric", "Then name of the metric to use", false, "L2",
                                    &availableMetricsConstraints);

    ValueArg<double> mParameterArg("p", "metricparam", "metric parameter", false, 2.0, "double");

    SwitchArg weighArg("w", "weigh", "weigh the votes by 1/(distance^2)");

    cmdLine.add(kArg);
    cmdLine.add(metricArg);
    cmdLine.add(mParameterArg);

    cmdLine.add(weighArg);

    std::vector<std::string> argv;
    std::string paramsWithName = std::string("progname ") + params;
    boost::split(argv, paramsWithName, boost::is_any_of(" "));
    cmdLine.setExceptionHandling(false);
    bool rebuildTree = false;
    try
    {

      cmdLine.parse(argv);

      rebuildTree = metricArg.getValue() != this->m|| this->mParam
      != mParameterArg.getValue() || this->w != weighArg.isSet()
      || this->classifier == NULL;

      this->k
      =kArg.getValue();
      this->m = metricArg.getValue();
      this->mParam = mParameterArg.getValue();
      this->w = weighArg.isSet();
    }
    catch (TCLAP::ArgException& ae)
    {

      throw ICFException()
          << invalid_argument(ae.error() + std::string(" ") + ae.argId() + std::string(" ") + ae.typeDescription());
    }
    catch (TCLAP::ExitException& ee)
    {
      throw ICFException() << invalid_state_error("TCLAP threw unexpected exception");
    }

    return rebuildTree;
  }

template class KNNClassifier<double> ;
template class KNNClassifier<float> ;

}
