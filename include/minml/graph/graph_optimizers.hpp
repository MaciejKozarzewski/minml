/*
 * graph_optimizers.hpp
 *
 *  Created on: Feb 28, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef LIBML_GRAPH_OPTIMIZATION_GRAPH_OPTIMIZERS_HPP_
#define LIBML_GRAPH_OPTIMIZATION_GRAPH_OPTIMIZERS_HPP_

namespace ml
{
	class Graph;
}

namespace ml
{
	class FoldBatchNorm
	{
		public:
			bool optimize(Graph &graph) const;
	};

	class FoldAdd
	{
		public:
			bool optimize(Graph &graph) const;
	};

	class FoldGelu
	{
		public:
			bool optimize(Graph &graph) const;
	};

} /* namespace ml */

#endif /* LIBML_GRAPH_OPTIMIZATION_GRAPH_OPTIMIZERS_HPP_ */
