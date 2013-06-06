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
#include <icf_core/classifiers/SVMClassifier.h>
#include <ros/ros.h>
using namespace std;
namespace icf
{

/** \brief svm print function*/
void printFunc(const char * msg)
{
}

SVMClassifier::SVMClassifier(const std::string& commandLine) :
    model(NULL), labelMap(NULL), problem(NULL), nrThreads(4)
{
  svm_set_print_string_function(printFunc);
  // default values
  param.svm_type = 0; //fixed to C-SVC
  param.kernel_type = RBF;
  param.degree = 3;
  param.gamma = 0; // 1/num_features
  param.coef0 = 0;
  param.nu = 0.5;
  param.cache_size = 100;
  param.C = 1;
  param.eps = 1e-3;
  param.p = 0.1;
  param.shrinking = 1;
  param.probability = 0;
  param.nr_weight = 0;
  param.weight_label = NULL;
  param.weight = NULL;
  cross_validation = 0;
  parse_command_line(commandLine);
  std::stringstream ss;
  ss << rand();
  ds_prefix = ss.str();

}

SVMClassifier::~SVMClassifier()
{
  if (model != NULL)
  {
    svm_free_and_destroy_model(&model);
  }
  if (this->problem != NULL)
  {
    delete[] this->problem->y;
    //delete [] this->problem->x[0];
    delete[] this->problem->x;
  }
  if (this->labelMap != NULL)
  {
    delete this->labelMap;
  }
}

void SVMClassifier::addTrainingData(std::string data)
{
  this->trainingData = data;
}

void SVMClassifier::addData(std::string data)
{
  this->classificationData = data;
}

void SVMClassifier::addEvaluationData(std::string evaluationData)
{
  this->evaluationData = evaluationData;
}

void SVMClassifier::cvThread(std::queue<std::pair<double, double> >& workQueue, double& best_error, double& best_c,
                             double& best_g, boost::mutex& queueMutex, boost::mutex& bestParamsMutex,
                             svm_problem& problem)
{
  using namespace std;
  boost::thread::id id = boost::this_thread::get_id();

  double * target = new double[problem.l];
  svm_parameter param = this->param;

  while (true)
  {

    queueMutex.lock();

    if (!workQueue.empty())
    {

      std::pair<double, double> workItem = workQueue.front();
      workQueue.pop();
      queueMutex.unlock();
      param.C = workItem.first;
      param.gamma = workItem.second;
      std::cout<<"Trying C="<<param.C<<" gamma="<< param.gamma<<" "<<std::endl;
    }
    else
    {

      queueMutex.unlock();
      return;
    }

    svm_cross_validation(&problem, &param, nr_fold, target);

    double total_error = 0.0;
    for (int i = 0; i < problem.l; i++)
    {
      double y = problem.y[i];
      double v = target[i];
      total_error += v == y ? 0 : 1;
    }
    total_error /= (double)problem.l;

    bestParamsMutex.lock();
    printf("total_error: %f\n", total_error);
    if (total_error < best_error)
    {
      printf("New best\n");
      best_error = total_error;
      best_c = param.C;
      best_g = param.gamma;
    }
    bestParamsMutex.unlock();
  }

}

struct cvThreadCallable
{
public:
  cvThreadCallable(SVMClassifier* _this, std::queue<std::pair<double, double> >& workQueue, double& best_error,
                   double& best_c, double& best_g, boost::mutex& queueMutex, boost::mutex& bestParamsMutex,
                   svm_problem& problem) :
      _this(_this), workQueue(workQueue), best_error(best_error), best_c(best_c), best_g(best_g), queueMutex(
          queueMutex), bestParamsMutex(bestParamsMutex), problem(problem)
  {

  }

  void operator()()
  {
    _this->cvThread(workQueue, best_error, best_c, best_g, queueMutex, bestParamsMutex, problem);
  }

  SVMClassifier * _this;
  std::queue<std::pair<double, double> >& workQueue;
  double& best_error;
  double& best_c;
  double& best_g;
  boost::mutex& queueMutex;
  boost::mutex& bestParamsMutex;
  svm_problem& problem;
};

int SVMClassifier::buildModel(std::string params)
{

  clock_t start = clock();
  parse_command_line(params);
  if (this->problem != NULL)
  {
    delete[] this->problem->y;
    //delete [] this->problem->x[0];
    delete[] this->problem->x;
  }
  else
  {
    problem = new svm_problem();

  }

  if (this->trainingData != "")
  {
    DS *ds = new DataSet<double>("/" + ds_prefix + "_train", (void*)trainingData.c_str(),
                                 trainingData.length() * sizeof(char), false);
    DS::Ptr ds_ptr(ds);
    this->trainDS = ds_ptr;
    //DS ds(name);
    //std::istringstream datastream(this->trainingData);
    //datastream >> ds;
    this->labelMap = new LabelMap(ds->getY());
    parse_data(this->trainingData, *problem, true);
  }
  else
  {
    this->labelMap = new LabelMap(this->trainDS->getY());
    parse_data(*this->trainDS, *problem, true);
  }
  if (this->cross_validation)
  {
    best_error = 1.0;
    double best_c = 1.0;
    double best_g = 1.0;
    std::queue<std::pair<double, double> > workQueue;
    boost::mutex queueMutex;
    boost::mutex bestParamsMutex;

    for (double c = cstart; c < cend; c += cstep)
    {
      for (double g = gstart; g < gend; g += gstep)
      {
        workQueue.push(std::pair<double, double>(pow(2.0, c), pow(2.0, g)));
      }
    }

    std::vector<boost::thread*> threads(this->nrThreads);

    for (int i = 0; i < this->nrThreads; i++)
    {
      std::cout<<"creating thread "<<i<<std::endl;
      cvThreadCallable cvThread(this, workQueue, best_error, best_c, best_g, queueMutex, bestParamsMutex, *problem);
      threads.push_back(new boost::thread(cvThread));
    }
    for (int i = 0; i < this->nrThreads; i++)
    {
      std::cout<<"waiting for thread "<<i<<std::endl;
      threads.back()->join();
      delete threads.back();
      threads.pop_back();
    }
    param.C = best_c;
    param.gamma = best_g;
    printf("Best error rate: %f\n", best_error);
    std::cout<<"Best params c , gamma "<<param.C<<" "<<param.gamma<<std::endl;
  }
  if (model != NULL)
  {
    svm_free_and_destroy_model(&model);
  }
  model = svm_train(problem, &param);
  this->secLastOp = (clock() - start) / (double)CLOCKS_PER_SEC;
  return 0;
}

std::map<int, int> getLabelPermutation(int * labels, int nr_of_classes)
{
  std::map<int, int> perm;
  for (int i = 0; i < nr_of_classes; i++)
  {
    int nr_smaller = 0;
    int lbl = labels[i];
    for (int j = 0; j < nr_of_classes; j++)
    {
      if (lbl > labels[j])
      {
        nr_smaller++;
      }
    }
    perm[i] = nr_smaller;
  }
  return perm;
}

std::string SVMClassifier::classify()
{
  clock_t start = clock();
  svm_problem problem;
  if (this->classificationData != "")
  {
    parse_data(this->classificationData, problem, true);
  }
  else
  {
    parse_data(*this->classifyDS, problem, false);
  }

  double * prob_estimates = NULL;
  bool doProbEstimate = false;
  int nr_of_classes = svm_get_nr_class(model);
  std::map<int, int> labelPermutation;
  DataSet<double>::MatrixPtr confidencesPtr;
  confidencesPtr = DataSet<double>::MatrixPtr((new DataSet<double>::Matrix(problem.l, nr_of_classes)));
  //ensures that prob estimates are stored for classes in ascending order
  int * labels = new int[nr_of_classes];
  svm_get_labels(model, labels);
  labelPermutation = getLabelPermutation(labels, nr_of_classes);
  if (doProbEstimate)
  {
    prob_estimates = new double[nr_of_classes];
  }

  boost::shared_ptr<ClassificationResult> result(new ClassificationResult(*this->labelMap));
  std::stringstream ss;
  for (int i = 0; i < problem.l; i++)
  {
    svm_node *x = problem.x[i];
    if (doProbEstimate)
    {
      result->add(svm_predict_probability(model, x, prob_estimates));
      for (int c = 0; c < nr_of_classes; c++)
      {
        (*confidencesPtr)(i, labelPermutation[c]) = prob_estimates[c];
      }
    }
    else
    {

      int prediction = svm_predict(model, x);
      int index = labelMap->mapToIndex(prediction);
      result->add(prediction);
      for (int c = 0; c < nr_of_classes; c++)
      {
        (*confidencesPtr)(i, c) = index == c ? 1 : 0;
      }
    }
  }
  if (doProbEstimate)
  {
    delete[] prob_estimates;
  }
  result->confidences = confidencesPtr;
  this->classificationResult = result;
  delete[] problem.y;
  delete[] problem.x[0];
  delete[] problem.x;
  ss << *result;
  this->secLastOp = (clock() - start) / (double)CLOCKS_PER_SEC;
  this->secPerInstanceClassification = secLastOp / problem.l;
  return ss.str();
}

void SVMClassifier::load(std::string filename)
{
  if (model != NULL)
  {
    svm_free_and_destroy_model(&model);
  }
  model = svm_load_model(filename.c_str());
  if (model == NULL)
  {
    throw ICFException() << invalid_state_error("Could not load model from file: " + filename);
  }
  std::string labelMapFileName = filename + ".cls";
  if (labelMap != NULL)
  {
    delete labelMap;
    labelMap = NULL;
  }
  labelMap = new LabelMap();
  if (!labelMap->load(labelMapFileName))
  {
    throw ICFException() << invalid_state_error("Could not load classes from file");
  }
  std::fstream in((filename + ".cm").c_str(), ios::in);
  if (in.good())
  {
    in.close();
    this->cm = boost::shared_ptr<ConfusionMatrix>(new ConfusionMatrix(filename + ".cm"));
  }
}

void SVMClassifier::save(std::string filename)
{
  if (model == NULL)
  {
    throw ICFException() << invalid_state_error("Model has to be built first");
  }
  std::string labelMapFileName = filename + ".cls";
  labelMap->save(labelMapFileName);
  svm_save_model(filename.c_str(), model);
  if (this->cm)
    this->cm->save(filename + ".cm");
}

/**
 * \brief parses data
 */
void SVMClassifier::parse_data(std::string data)
{

}

std::string SVMClassifier::executeService(std::string service, std::string params)
{
  if (service == "print_params")
  {
    stringstream ss;
    ss << "C " << model->param.C << endl;
    ss << "gamma " << model->param.gamma << endl;
    return ss.str();
  }
  else
  {
    return Classifier::executeService(service, params);
  }
}

std::string SVMClassifier::evaluate()
{
  clock_t start = clock();
  svm_problem problem;
  if (this->evaluationData != "")
  {
    DS * ds = new DataSet<double>("/" + ds_prefix + "_evaluate", (void*)evaluationData.c_str(),
                                  evaluationData.length() * sizeof(char), false);
    DS::Ptr ds_ptr(ds);
    this->evalDS = ds_ptr;

    this->labelMap = new LabelMap(ds->getY());
    parse_data(this->evaluationData, problem, true);
  }
  else
  {
    this->labelMap = new LabelMap(this->trainDS->getY());
    parse_data(*this->evalDS, problem, true);
  }
  boost::shared_ptr<std::vector<int> > classification(new std::vector<int>());
  boost::shared_ptr<std::vector<int> > groundTruth(new std::vector<int>());

  double * prob_estimates;
  bool doProbEstimate = false;
  int nr_of_classes = svm_get_nr_class(model);
  std::map<int, int> labelPermutation;
  DataSet<double>::MatrixPtr confidencesPtr;
  confidencesPtr = DS::MatrixPtr((new DS::Matrix(problem.l, nr_of_classes)));
  //ensures that prob estimates are stored for classes in ascending order
  int * labels = new int[nr_of_classes];
  svm_get_labels(model, labels);
  labelPermutation = getLabelPermutation(labels, nr_of_classes);

  if (doProbEstimate)
  {
    prob_estimates = new double[nr_of_classes];
  }

  for (int i = 0; i < problem.l; i++)
  {
    svm_node *x = problem.x[i];
    groundTruth->push_back(problem.y[i]);
    if (doProbEstimate)
    {
      classification->push_back(svm_predict_probability(model, x, prob_estimates));
      for (int c = 0; c < nr_of_classes; c++)
      {
        (*confidencesPtr)(i, labelPermutation[c]) = prob_estimates[c];
      }
    }
    else
    {
      int prediction = svm_predict(model, x);
      int index = labelMap->mapToIndex(prediction);
      classification->push_back(prediction);
      for (int c = 0; c < nr_of_classes; c++)
      {
        (*confidencesPtr)(i, c) = index == c ? 1.0 : 0.0;
      }
    }
  }
  if (doProbEstimate)
  {
    delete[] prob_estimates;
  }
  delete[] problem.y;
  delete[] problem.x[0];
  delete[] problem.x;
  std::cerr << classification->at(0) << " -> " << classification->at(1) << std::endl;
  boost::shared_ptr<EvaluationResult> result(new EvaluationResult(groundTruth, classification));
  result->setConfidence(confidencesPtr);
  this->cm = result->getConfusionMatrix();
  this->evalResult = result;
  std::stringstream ss;
  ss << *result;
  this->secLastOp = (clock() - start) / (double)CLOCKS_PER_SEC;
  this->secPerInstanceClassification = secLastOp / problem.l;
  return ss.str();
}
;

void SVMClassifier::parse_data(DataSet<double>& ds, svm_problem& problem, bool containsLabels)
{
  //DataSet<double>::Matrix& x = *ds.getX();
  DS::MatrixPtr x_ptr = ds.getX();
  DS::Matrix & x = *x_ptr;
  DS::MatrixPtr y_ptr;
  DS::Matrix y;
  if (containsLabels)
  {
    y_ptr = ds.getY();
    y = *y_ptr;
  }
  //DataSet<double>::Matrix& y = *ds.getY();
  //count number of elements
  std::string col;
  size_t elementCount = (x.cols() + 1) * (x.rows());
  problem.l = x.rows();

  //allocate memory
  problem.x = new svm_node*[problem.l];
  problem.y = new double[problem.l];
  svm_node* x_space = new svm_node[elementCount];
  //FILL PROBLEM STRUCTURE
  size_t offset = 0; //offset into x_space
  for (int r = 0; r < x.rows(); r++)
  {

    problem.x[r] = &x_space[offset];
    for (int c = 0; c < x.cols(); c++)
    {
      x_space[offset].index = c + 1;
      x_space[offset++].value = x(r, c);
    }
    x_space[offset].value = 0;
    x_space[offset++].index = -1;
    if (containsLabels)
      problem.y[r] = y(r, 0);
  }
}

/**
 * \brief
 */
void SVMClassifier::parse_data(const std::string& data, svm_problem& problem, bool containsLabels)
{
  DS * ds = new DataSet<double>("/" + ds_prefix + "_test", (void*)data.c_str(), data.length() * sizeof(char), false);
  parse_data(*ds, problem, containsLabels);
}

/**
 * \brief parses parameter string identical to libsvm format (except for the W, b and s options which are not supported yet)
 * \brief The v option also causes a grid search for the C and gamma parameters
 * \brief and has the format -v nr_folds cstart:cstep:cend gstart:gstep:gend where c* and g* are powers of 2
 */
void SVMClassifier::parse_command_line(const std::string& paramsString)
{

  if (paramsString == "")
    return;

  std::vector<std::string> paramTokens;
  boost::split(paramTokens, paramsString, boost::is_any_of(" "));

  // parse options
  for (std::vector<std::string>::const_iterator token = paramTokens.begin(); token < paramTokens.end(); token++)
  {
    const std::string& paramName = *token;

    if (token++ == paramTokens.end() || !boost::starts_with(paramName, "-"))
      throw ICFException()
          << invalid_argument("Invalid param format. Please see libsvm documentation for valid parameters.");

    const std::string& paramValue = *token;

    switch (paramName/*.c_str()*/[1])
    {
      //		case 's':
      //			param.svm_type = atoi(paramValue.c_str());
      //			break;
      case 'a':
        this->nrThreads = atoi(paramValue.c_str());
        if (this->nrThreads == 0)
        {
          this->nrThreads = 1;
          std::cerr << "No. of threads set to 1" << std::endl;
        }
        break;
      case 'b':
        param.probability = atoi(paramValue.c_str());
        break;
      case 't':
        param.kernel_type = atoi(paramValue.c_str());
        break;
      case 'd':
        param.degree = atoi(paramValue.c_str());
        break;
      case 'g':
        param.gamma = atof(paramValue.c_str());
        break;
      case 'r':
        param.coef0 = atof(paramValue.c_str());
        break;
      case 'n':
        param.nu = atof(paramValue.c_str());
        break;
      case 'm':
        param.cache_size = atof(paramValue.c_str());
        break;
      case 'c':
        param.C = atof(paramValue.c_str());
        break;
      case 'e':
        param.eps = atof(paramValue.c_str());
        break;
      case 'p':
        param.p = atof(paramValue.c_str());
        break;
      case 'h':
        param.shrinking = atoi(paramValue.c_str());
        break;
        //		case 'b':
        //			param.probability = atoi(paramValue.c_str());
        //			break;
      case 'v':
        cross_validation = true;
        nr_fold = atoi(paramValue.c_str());
        if (nr_fold < 2)
        {
          throw ICFException() << invalid_argument("number of folds must be >= 2");
        }
        //parse c range and step size
        if (token++ != paramTokens.end())
        {
          std::string crange = *token;
          std::vector<std::string> range;
          boost::split(range, crange, boost::is_any_of(":"));
          if (range.size() != 3)
          {
            throw ICFException() << invalid_argument("range for parameter c required: format start:step:end");
          }
          cstart = atof(range[0].c_str());
          cstep = atof(range[1].c_str());
          cend = atof(range[2].c_str());
        }
        else
        {
          throw ICFException() << invalid_argument("range for parameter c required: format start:step:end");
        }
        if (token++ != paramTokens.end())
        {
          std::string grange = *token;
          std::vector<std::string> range;
          boost::split(range, grange, boost::is_any_of(":"));
          if (range.size() != 3)
          {
            throw std::string("range for parameter g required: format start:step:end");
          }
          gstart = atof(range[0].c_str());
          gstep = atof(range[1].c_str());
          gend = atof(range[2].c_str());
        }
        else
        {
          throw ICFException() << invalid_argument("range for parameter g required: format start:step:end");
        }

        break;
      default:
        throw ICFException() << invalid_state_error("Unknown option " + paramName);
    }
  }
}
}
