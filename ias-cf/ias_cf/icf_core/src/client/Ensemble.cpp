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

#include <icf_core/client/Ensemble.h>

namespace icf{

Ensemble::Ensemble()
{

}

Ensemble::~Ensemble()
{

}

std::vector<CBPtr>& Ensemble::getClassifiers()
{
	return classifiers;
}

ClassificationResult Ensemble::classify()
{
	lastOperationExceptions.clear();
	std::vector<boost::tuple<CBPtr, ClassificationResult> > classificationResults;
	for (std::vector<CBPtr>::iterator iter = classifiers.begin(); iter
			!= classifiers.end(); iter++) {
		try {
			//TODO Parallelize this
			classificationResults.push_back(boost::make_tuple<CBPtr,
					ClassificationResult>(*iter, (*iter)->classify()));
		} catch (ICFException & e) {
			lastOperationExceptions.push_back(e);
		}
	}

	return classifyInternal(classificationResults);

}


EvaluationResult Ensemble::evaluate(DS::MatrixPtr groundTruth)
{
	lastOperationExceptions.clear();
	std::vector<boost::tuple<CBPtr, EvaluationResult> > evaluationResults;
	for (std::vector<CBPtr>::iterator iter = classifiers.begin(); iter
			!= classifiers.end(); iter++) {
		try {
			//TODO Parallelize this
			evaluationResults.push_back(boost::make_tuple<CBPtr,
					EvaluationResult>(*iter, (*iter)->evaluate()));
		} catch (ICFException & e) {
			lastOperationExceptions.push_back(e);
		}
	}
	EvaluationResult thisEvaluation = evaluateInternal(evaluationResults,groundTruth);

	return thisEvaluation;

}


};

