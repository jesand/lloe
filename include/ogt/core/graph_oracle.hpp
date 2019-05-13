/// @file  graph_oracle.hpp
/// @brief Declarations related to comparison oracles on graphs.

#pragma once
#ifndef OGT_CORE_GRAPH_ORACLE_HPP
#define OGT_CORE_GRAPH_ORACLE_HPP

#include <ogt/config.hpp>
#include <ogt/core/oracle.hpp>
#include <iostream>
#include <memory>

namespace OGT_NAMESPACE {
namespace core {

/// Define our graph datatype
struct Graph {

	/// Virtual destructor
	virtual ~Graph() = default;

	/// Load the graph from an edge list, up to nObj vertices
	static std::shared_ptr<Graph> CreateFromEdgeList(std::istream& is, size_t nObj=0);

	/// Get the number of nodes in the graph
	virtual size_t vertices() const = 0;

	/// Get the number of edges in the graph
	virtual size_t edges() const = 0;

	/// Get the real identifier of the vertex with the given index
	virtual int vertexId(size_t node) const = 0;

	/// Get the shortest path length between two nodes.
	virtual size_t shortestPathLength(size_t source, size_t target) = 0;

	/// Ask how much time was taken to compute graph distances
	virtual double graphDistanceTime() const = 0;
};

/// Get an oracle which uses shortest path distance on a graph.
std::shared_ptr<Oracle> createShortestPathOracle(std::shared_ptr<Graph> g);

} // end namespace core
} // end namespace OGT_NAMESPACE
#endif /* OGT_CORE_GRAPH_ORACLE_HPP */
