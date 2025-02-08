/*
 * swa_utils.hpp
 *
 *  Created on: Feb 8, 2025
 *      Author: Maciej Kozarzewski
 */

#ifndef MINML_GRAPH_SWA_UTILS_HPP_
#define MINML_GRAPH_SWA_UTILS_HPP_

namespace ml
{
	class Graph;
}

namespace ml
{
	// computes averaged = alpha * model + beta * averaged
	void averageModelWeights(float alpha, const Graph &model, float beta, Graph &averaged);
	void updateBatchNormStats(Graph &model);
} /* namespace ml */

#endif /* MINML_GRAPH_SWA_UTILS_HPP_ */
