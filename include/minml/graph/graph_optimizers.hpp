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
	class CalibrationTable;
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

	class Quantize
	{
		public:
			bool optimize(Graph &graph, int bits) const;
	};

	class FuseConvBlock
	{
		public:
			bool optimize(Graph &graph) const;
	};

	class FuseSEBlock
	{
		public:
			bool optimize(Graph &graph) const;
	};

} /* namespace ml */

#endif /* LIBML_GRAPH_OPTIMIZATION_GRAPH_OPTIMIZERS_HPP_ */
