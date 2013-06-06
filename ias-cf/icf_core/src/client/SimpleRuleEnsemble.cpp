/*
 * SimpleRuleEnsemble.cpp
 *
 *  Created on: 06.10.2011
 *      Author: florian
 */

#include <icf_core/client/SimpleRuleEnsemble.h>
#include <boost/tuple/tuple.hpp>
namespace icf {

SimpleRuleEnsemble::SimpleRuleEnsemble(AggregationMode am,
		AggregatedInformation ai) :
	am(am), ai(ai) {

}

SimpleRuleEnsemble::~SimpleRuleEnsemble() {

}

Matrix * calculateScores(
		std::vector<boost::tuple<CBPtr, EvaluationResult> >& classificationResults,
		SimpleRuleEnsemble::AggregatedInformation ai,
		SimpleRuleEnsemble::AggregationMode am) {
	//TODO: Modify this and the superclass to only evaluate the minimum number of classifiers to reach a decision;
	//i.e. if a a decision can not be changed by further evaluations because it's confidence is very high,
	//then don't evaluate further classifiers
	if (classificationResults.size() == 0) {
		throw ICFException() << invalid_argument(
				"classificationResults is empty");
	}
	EvaluationResult result = (*classificationResults.begin()).get<1> ();
	int noClasses = result.confidences->cols();
	int noRows = result.confidences->rows();

	DS::Matrix * m = NULL;
	m = new Matrix();
	m->setConstant(noRows, noClasses, am == SimpleRuleEnsemble::PRODUCT ? 1.0
			: 0.0);

	DS::Matrix& mr = *m;
	DS::Matrix pc;
	pc.setConstant(mr.rows(), mr.cols(), 0.01);
	for (std::vector<boost::tuple<CBPtr, EvaluationResult> >::iterator iter =
			classificationResults.begin(); iter != classificationResults.end(); iter++) {
		CBPtr classifier = iter->get<0> ();
		EvaluationResult result = iter->get<1> ();

		if (ai == SimpleRuleEnsemble::CONFIDENCE_ACCURACY || ai
				== SimpleRuleEnsemble::CONFIDENCE) {
			if (am == SimpleRuleEnsemble::SUM) {
				mr += *result.confidences;
			} else if (am == SimpleRuleEnsemble::PRODUCT) {
				mr.array() *= (result.confidences->array() + pc.array());
			}
		}

		if (ai == SimpleRuleEnsemble::CONFIDENCE_ACCURACY || ai
				== SimpleRuleEnsemble::ACCURACY) {
			DS::Matrix& results = *result.results;
			result.getConfusionMatrix()->getLabelMap().mapToIndex(results);
			classifier->getConfusionMatrix()->normalizeRows();
			DS::Matrix cm = classifier->getConfusionMatrix()->getCM();
			for (int r = 0; r < noRows; r++) {
				for (int c = 0; c < noClasses; c++) {
					double p = cm(c, results(r, 0));
					if (am == SimpleRuleEnsemble::PRODUCT) {
						mr(r, c) *= (0.01 + p);
					} else if (am == SimpleRuleEnsemble::SUM) {
						mr(r, c) += p;
					}
				}
			}
		}
	}
	for (int i = 0; i < mr.rows(); i++) {
		mr.row(i) /= mr.row(i).norm();
	}
	return m;
}

Matrix * calculateScores(
		std::vector<boost::tuple<CBPtr, ClassificationResult> >& classificationResults,
		SimpleRuleEnsemble::AggregatedInformation ai,
		SimpleRuleEnsemble::AggregationMode am) {
	//TODO: Modify this and the superclass to only evaluate the minimum number of classifiers to reach a decision;
	//i.e. if a a decision can not be changed by further evaluations because it's confidence is very high,
	//then don't evaluate further classifiers
	if (classificationResults.size() == 0) {
		throw ICFException() << invalid_argument(
				"No classification results were passed to the ensemble");
	}
	ClassificationResult result = (*classificationResults.begin()).get<1> ();
	int noClasses = result.confidences->cols();
	int noRows = result.confidences->rows();

	DS::Matrix * m = NULL;
	m = new Matrix();
	m->setConstant(noRows, noClasses, am == SimpleRuleEnsemble::PRODUCT ? 1.0
			: 0.0);

	DS::Matrix& mr = *m;
	DS::Matrix pc;
	pc.setConstant(mr.rows(), mr.cols(), 0.01);
	for (std::vector<boost::tuple<CBPtr, ClassificationResult> >::iterator
			iter = classificationResults.begin(); iter
			!= classificationResults.end(); iter++) {
		CBPtr classifier = iter->get<0> ();
		ClassificationResult result = iter->get<1> ();

		if (ai == SimpleRuleEnsemble::CONFIDENCE_ACCURACY || ai
				== SimpleRuleEnsemble::CONFIDENCE) {
			if (am == SimpleRuleEnsemble::SUM) {
				mr += *result.confidences;
			} else if (am == SimpleRuleEnsemble::PRODUCT) {
				mr.array() *= (result.confidences->array() + pc.array());
			}
		}

		if (ai == SimpleRuleEnsemble::CONFIDENCE_ACCURACY || ai
				== SimpleRuleEnsemble::ACCURACY) {
			std::vector<int>& results = *result.results;
			result.labelMap.mapToIndex(results);
			classifier->getConfusionMatrix()->normalizeRows();
			DS::Matrix cm = classifier->getConfusionMatrix()->getCM();
			for (int r = 0; r < noRows; r++) {
				for (int c = 0; c < noClasses; c++) {
					double p = cm(c, results[r]);
					if (am == SimpleRuleEnsemble::PRODUCT) {
						mr(r, c) *= (0.01 + p);
					} else if (am == SimpleRuleEnsemble::SUM) {
						mr(r, c) += p;
					}
				}
			}
		}
	}
	for (int i = 0; i < mr.rows(); i++) {
		mr.row(i) /= mr.row(i).norm();
	}
	return m;
}

std::vector<int> maxIndicesRow(const DS::Matrix& m) {
	std::vector<int> maxKeys;
	if (m.rows() == 0)
		return maxKeys;
	maxKeys.push_back(0);
	double maxVal = m(0, 0);

	for (int i = 1; i < m.cols(); i++) {
		if (m(0, i) > maxVal) {
			maxKeys.clear();
			maxKeys.push_back(i);
			maxVal = m(0, i);
		} else if (m(0, i) == maxVal) {
			maxKeys.push_back(i);
		}
	}
	return maxKeys;
}

/**
 * Sample a position between start and end
 */
template<class T>
T sample_uniform(T start, T end) {
	int range = end - start;
	int i = rand() % range;
	return start + i;
}

std::vector<int> * maxIndices(DS::Matrix * scores) {
	std::vector<int>* maxIndicesVec = new std::vector<int>();
	for (int r = 0; r < scores->rows(); r++) {
		Matrix m = scores->row(r);//TODO: get rid of this temporary object
		std::vector<int> allMaxIndices = maxIndicesRow(m);
		maxIndicesVec->push_back(*sample_uniform(allMaxIndices.begin(),
				allMaxIndices.end()));
	}
	return maxIndicesVec;
}

std::vector<int> * maxLabels(std::vector<int> * indices, LabelMap map) {
	map.mapToLabels(*indices);
	return indices;
}

ClassificationResult SimpleRuleEnsemble::classifyInternal(std::vector<
		boost::tuple<CBPtr, ClassificationResult> >& classificationResults) {
	DS::Matrix * scores = calculateScores(classificationResults, ai, am);
	ClassificationResult result(
			classificationResults.begin()->get<1> ().labelMap,
			boost::shared_ptr<std::vector<int> >(maxLabels(maxIndices(scores),
					classificationResults.begin()->get<1> ().labelMap)));
	result.confidences.reset(scores);
	return result;
}

EvaluationResult SimpleRuleEnsemble::evaluateInternal(std::vector<boost::tuple<
		CBPtr, EvaluationResult> >& evaluationResults,DS::MatrixPtr groundTruth) {

	DS::Matrix * scores = calculateScores(evaluationResults, ai, am);
	LabelMap
			labelMap =
					evaluationResults.begin()->get<1> ().getConfusionMatrix()->getLabelMap();
	std::vector<int> * labels = maxLabels(maxIndices(scores), labelMap);
	EvaluationResult result(groundTruth, boost::shared_ptr<std::vector<int> >(
			labels));
	result.confidences.reset(scores);
	return result;
}

}
