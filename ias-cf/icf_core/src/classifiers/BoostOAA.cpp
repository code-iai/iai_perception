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

#include <icf_core/classifiers/BoostOAA.h>

using namespace cv;
using namespace std;
using namespace TCLAP;
using namespace boost;

namespace icf
{

BoostOAA::BoostOAA(std::string params, int threadCount) :
    threadCount(threadCount)
{
  parse_parameters(params);
  std::stringstream ss;
  ss << rand();
  this->ds_prefix = ss.str();
}

BoostOAA::~BoostOAA()
{

}

void BoostOAA::parse_parameters(std::string params)
{

  CmdLine cmdLine(" ", ' ', " ", false);

  std::vector<int> availableBoostTypes;
  availableBoostTypes.push_back(0);
  availableBoostTypes.push_back(1);
  availableBoostTypes.push_back(2);
  availableBoostTypes.push_back(3);

  ValuesConstraint<int> availableBoostTypesConstraints(availableBoostTypes);
  ValueArg<int> boostTypeArg("t", "type", "0:DISCRETE, 1:REAL, 2:LOGIT, 3:GENTLE", false, 2,
                             &availableBoostTypesConstraints);

  ValueArg<unsigned int> weakCountArg("", "weak_count", "weak_count", false, 50, "unsigned int");
  ValueArg<double> weightTrimRateArg("", "weight_trim_rate", "weight_trim_rate", false, 0.0, "double");
  ValueArg<unsigned int> maxDepthArg("", "max_depth", "max_depth", false, 1, "unsinged int");
  ValueArg<bool> useSurrogatesArg("", "use_surrogates", "use_surrogates", false, false, "bool");

  SwitchArg squashingArg("s", "squash", "use squashing function", false);

  cmdLine.add(boostTypeArg);
  cmdLine.add(weakCountArg);
  cmdLine.add(weightTrimRateArg);
  cmdLine.add(maxDepthArg);
  cmdLine.add(useSurrogatesArg);
  cmdLine.add(squashingArg);

  vector<string> argv;
  string paramsWithName = string("progname ") + params;
  boost::split(argv, paramsWithName, boost::is_any_of(" "));
  cmdLine.setExceptionHandling(false);

  try
  {

    cmdLine.parse(argv);

    this->params = CvBoostParams(boostTypeArg.getValue(), weakCountArg.getValue(), weightTrimRateArg.getValue(),
                                 maxDepthArg.getValue(), useSurrogatesArg.getValue(), NULL);
    this->useSquashingFunction = squashingArg.getValue();
  }
  catch (TCLAP::ArgException& ae)
  {

    throw ICFException()
        << invalid_argument(ae.error() + std::string(" ") + ae.argId() + std::string(" ") + ae.typeDescription());
  }
  catch (TCLAP::ExitException& ee)
  {
    throw ICFException() << invalid_argument("TCLAP command line parser threw unexpected exception");
  }

}

/**
 *\brief method for adding training data
 *\param data to be int BoostClassifier::erperted and added
 */
void BoostOAA::addTrainingData(std::string data)
{

  DS * ds = new DataSet<double>(ds_prefix + "_train", (void*)data.c_str(), data.length() * sizeof(char), false);
  DS::Ptr ds_ptr(ds);
  this->trainDS = ds_ptr;
}

void BoostOAA::addData(std::string data)
{
  DS * ds = new DataSet<double>(ds_prefix + "_classify", (void*)data.c_str(), data.length() * sizeof(char), false);
  DS::Ptr ds_ptr(ds);
  this->classifyDS = ds_ptr;
}

void BoostOAA::addEvaluationData(std::string data)
{

  DS * ds = new DataSet<double>(ds_prefix + "_evaluate", (void*)data.c_str(), data.length() * sizeof(char), false);
  DS::Ptr ds_ptr(ds);
  this->evalDS = ds_ptr;

}

void relabel(cv::Mat& labels, float cls)
{
  for (int i = 0; i < labels.rows; i++)
  {
    //std::cout<<labels.at<float>(i,0)<<std::endl;
    if (labels.at<float>(i, 0) == cls)
    {
      labels.at<float>(i, 0) = 1;
    }
    else
    {
      labels.at<float>(i, 0) = -1;
    }
    //std::cout<<"\t"<<labels.at<float>(i,0)<<std::endl;
  }
}

struct TrainingThread
{
  int clazz;
  Mat& x;
  Mat& y;
  mutex& m;
  mutex& c;
  std::vector<std::pair<int, boost::shared_ptr<CvBoost> > >& classifiers;
  CvBoostParams& params;
  int& threads;
  bool update;

  TrainingThread(int clazz, Mat&x, Mat&y, mutex& m, mutex& c,
                 std::vector<std::pair<int, boost::shared_ptr<CvBoost> > >& classifiers, CvBoostParams& params,
                 int & threads, bool update) :
      clazz(clazz), x(x), y(y), m(m), c(c), classifiers(classifiers), params(params), threads(threads), update(update)
  {

  }

  void operator()()
  {
    c.lock();
    //std::cout << "Training class " << clazz << endl;
    c.unlock();

    Mat _y = y.clone();

    relabel(_y, clazz);

    CvBoost * classifier = new CvBoost();

    classifier->train(x, CV_ROW_SAMPLE, _y, Mat(), Mat(), Mat(), Mat(), params, update);

    c.lock();
    classifiers.push_back(std::pair<int, boost::shared_ptr<CvBoost> >(clazz, boost::shared_ptr<CvBoost>(classifier)));
    threads++;
    c.unlock();
  }
};

/**
 *\brief
 */
int BoostOAA::buildModel(std::string params)
{

  clock_t start = clock();

  bool update = false;
  if (params != "" && !boost::starts_with(params, "update"))
  {
    parse_parameters(params);
  }
  else if (boost::starts_with(params, "update"))
  {
    vector<string> result;
    boost::split(result, params, boost::is_any_of(" "));
    if (result.size() != 2)
    {
      throw ICFException() << invalid_argument("Syntax: update #number_of_weak_learners");
    }
    update = true;
    this->params.weak_count = atoi(result.at(1).c_str());
  }
  if (!update)
  {
    classifiers.clear();
  }

  Mat x = this->trainDS->getAsCvMat<float>("/x");
  Mat y = this->trainDS->getAsCvMat<float>("/y");

  set<int> classes = countClasses(y);
  labelMap = LabelMap(trainDS->getY());
  int threads = threadCount;
  for (set<int>::iterator iter = classes.begin(); iter != classes.end(); iter++)
  {
    bool waiting = true;
    do
    {

      m.lock();
      if (threads > 0)
      {
        threads--;
        boost::thread(TrainingThread(*iter, x, y, m, c, classifiers, this->params, threads, update));
        waiting = false;
      }
      m.unlock();
      if (waiting)
        boost::this_thread::yield();
    } while (waiting);

  }
  bool stop = false;
  do
  {
    m.lock();
    stop = (threads == threadCount);
    m.unlock();
    boost::this_thread::yield();
  } while (!stop);
  this->secLastOp = (clock() - start) / (double)CLOCKS_PER_SEC;
  return 0;
}

/**
 *\brief classifying/adding test data use this function when you don't have ground truth
 *\return result for given test data
 *\param data to be tested
 */
std::string BoostOAA::classify()
{
  clock_t start = clock();
  Mat x = this->classifyDS->getAsCvMat<float>("/x");

  boost::shared_ptr<ClassificationResult> result(new ClassificationResult(labelMap));
  DS::MatrixPtr confidencesPtr(new DS::Matrix(x.rows, classifiers.size()));
  int prediction = 0;
  double best_p = -DBL_MAX;
  for (int s = 0; s < x.rows; s++)
  {
    int i = 0;
    for (vector<pair<int, boost::shared_ptr<CvBoost> > >::iterator iter = classifiers.begin();
        iter != classifiers.end(); iter++, i++)
    {
      Mat sample = x.row(s);
      double p = iter->second->predict(sample, Mat(1, x.cols, CV_8UC1), Range::all(), false, true);
      (*confidencesPtr)(s, i) = p;
      if (p > best_p)
      {
        prediction = iter->first;
        best_p = p;
      }
    }
    best_p = -DBL_MAX;
    result->add(prediction);
    //use softmax function on activations
    if (useSquashingFunction)
    {
      double sfconf[confidencesPtr->cols()];
      for (int i = 0; i < confidencesPtr->cols(); i++)
      {
        double pi = (*confidencesPtr)(s, i);
        double denominator = 1.0;
        for (int c = 0; c < confidencesPtr->cols(); c++)
        {
          if (c == i)
            continue;
          denominator += exp((*confidencesPtr)(s, c) - pi);
        }
        sfconf[i] = 1.0 / denominator;
      }
      for (int i = 0; i < confidencesPtr->cols(); i++)
      {
        (*confidencesPtr)(s, i) = sfconf[i];
      }
    }
  }
  result->confidences = confidencesPtr;
  this->classificationResult = result;
  stringstream out;
  out << *result;
  this->secLastOp = (clock() - start) / (double)CLOCKS_PER_SEC;
  this->secPerInstanceClassification = secLastOp / x.rows;
  return out.str();
}

/**
 *\brief basically the same as classify, use it when you have ground truth so confusion matrix can be created
 */
std::string BoostOAA::evaluate()
{
  clock_t start = clock();
  DS::MatrixPtr groundTruth = this->evalDS->getFeatureMatrix("/y");
  Mat x = this->evalDS->getAsCvMat<float>("/x");

  boost::shared_ptr<vector<int> > predictions(new vector<int>());

  DS::MatrixPtr confidencesPtr(new DS::Matrix(x.rows, classifiers.size()));

  int prediction = 0;
  double best_p = -DBL_MAX;

  for (int s = 0; s < x.rows; s++)
  {
    int i = 0;

    for (vector<pair<int, boost::shared_ptr<CvBoost> > >::iterator iter = classifiers.begin();
        iter != classifiers.end(); iter++, i++)
    {
      Mat sample = x.row(s);
      //std::cout<<"Prediction time"<<std::endl;
      //std::cout<<sample.cols<<std::endl;
      //std::cout<<sample.rows<<std::endl;
      double p = iter->second->predict(sample, Mat(1, x.cols, CV_8UC1), Range::all(), false, true);
      (*confidencesPtr)(s, i) = p;
      if (p > best_p)
      {
        prediction = iter->first;
        best_p = p;
      }
    }
    best_p = -DBL_MAX;
    predictions->push_back(prediction);
    prediction = 0;
    if (useSquashingFunction)
    {
      //use softmax function
      double sfconf[confidencesPtr->cols()];
      for (int i = 0; i < confidencesPtr->cols(); i++)
      {
        double pi = (*confidencesPtr)(s, i);
        double denominator = 1.0;
        for (int c = 0; c < confidencesPtr->cols(); c++)
        {
          if (c == i)
            continue;
          denominator += exp((*confidencesPtr)(s, c) - pi);
        }
        sfconf[i] = 1.0 / denominator;
      }
      for (int i = 0; i < confidencesPtr->cols(); i++)
      {
        (*confidencesPtr)(s, i) = sfconf[i];
      }
    }
  }
  boost::shared_ptr<EvaluationResult> result(new EvaluationResult(groundTruth, predictions));
  result->setConfidence(confidencesPtr);
  this->cm = result->getConfusionMatrix();
  this->evalResult = result;
  stringstream out;
  out << *result;
  this->secLastOp = (clock() - start) / (double)CLOCKS_PER_SEC;
  this->secPerInstanceClassification = secLastOp / x.rows;
  return out.str();

}

/**
 * \brief save the model. Throw a std::string if something goes wrong.
 */
void BoostOAA::save(std::string filename)
{

  string fileNameCls = filename + string(".cls");
  fstream cls(fileNameCls.c_str(), ios_base::out | ios_base::trunc);
  for (vector<pair<int, boost::shared_ptr<CvBoost> > >::iterator iter = classifiers.begin(); iter != classifiers.end();
      iter++)
  {
    if (cls.fail())
    {
      throw ICFException() << invalid_state_error("Can't open file " + filename + " for saving adaboost classifier");
    }
    stringstream ss;
    ss << "_" << iter->first;
    iter->second->save((filename + ss.str()).c_str(), ss.str().c_str());
    cls << iter->first;
    if (iter + 1 != classifiers.end())
      cls << " " << endl;
  }
  cls.close();
  if (this->cm)
    this->cm->save(filename + ".cm");
}

/**
 * \brief load the model. Throw a std::string if something goes wrong.
 */
void BoostOAA::load(std::string filename)
{
  string fileNameCls = filename + string(".cls");
  fstream cls(fileNameCls.c_str(), ios_base::in);
  while (!cls.eof())
  {
    if (cls.fail())
    {
      throw ICFException() << invalid_state_error("Can't open file " + filename + " for loading adaboost classifier");
    }
    int clazz;
    cls >> clazz;
    stringstream ss;
    ss << "_" << clazz;
    pair<int, boost::shared_ptr<CvBoost> > classifier(clazz, boost::shared_ptr<CvBoost>(new CvBoost()));
    //cout << "Loading classifier for class: " << clazz << endl;
    classifier.second->load((filename + ss.str()).c_str(), ss.str().c_str());
    classifiers.push_back(classifier);
  }
  labelMap.load(fileNameCls);
  cls.close();
  std::ifstream test((filename + ".cm").c_str(), ios::in);
  if (test.good())
    this->cm = boost::shared_ptr<ConfusionMatrix>(new ConfusionMatrix(filename + ".cm"));
}

void BoostOAA::relabel(cv::Mat& labels, float cls)
{
  for (int i = 0; i < labels.rows; i++)
  {
    if (labels.at<float>(i, 0) == cls)
    {
      labels.at<float>(i, 0) = 1;
    }
    else
    {
      labels.at<float>(i, 0) = -1;
    }
  }
}

set<int> BoostOAA::countClasses(cv::Mat labels)
{
  set<int> cls;
  for (int i = 0; i < labels.rows; i++)
  {
    cls.insert(round(labels.at<float>(i, 0)));
  }
  return cls;
}

}
