/// @file  graph.hpp
/// @brief Graph algorithms.

#pragma once
#ifndef OGT_UTIL_GRAPH_HPP
#define OGT_UTIL_GRAPH_HPP

#include <ogt/config.hpp>
#include <map>
#include <vector>

namespace OGT_NAMESPACE {
namespace util {

/// A directed graph with weighted edges.
struct DiGraph {

	/// Default ctor
	DiGraph();

	/// Initialize a disconnected graph.
	DiGraph(const std::vector<size_t>& nodes);

	/// A node in the graph.
	struct Edges {
		Edges(size_t data) : data(data), inDegree(0), outDegree(0) {}

		const size_t data;
		double inDegree;
		double outDegree;
		std::map<size_t, double> next;
		std::map<size_t, double> prev;
	};

	/// Get the number of nodes in the graph.
	size_t numNodes() const;

	/// Get the number of edges in the graph.
	size_t numEdges() const;

	/// Ask whether there is a node with the given identifier.
	bool hasNode(size_t node) const;

	/// Ask whether there is an edge between the given nodes.
	bool hasEdge(size_t from, size_t to) const;

	/// Get the weight for an edge, or 0 if there is no such edge.
	double edgeWeight(size_t from, size_t to) const;

	/// Add a node to the graph.
	void addNode(size_t node);

	/// Remove a node from the graph.
	void removeNode(size_t node);

	/// Get a node's in-degree (sum of weights of incoming edges).
	double nodeInDegree(size_t node) const;

	/// Get a node's out-degree (sum of weights of outgoing edges).
	double nodeOutDegree(size_t node) const;

	/// Add an edge to the graph.
	void addEdge(size_t from, size_t to, double weight);

	/// Remove an edge from the graph.
	void removeEdge(size_t from, size_t to);

	/// Get a node with minimal in-degree, or SIZE_MAX if there are no nodes.
	size_t minInDegreeNode();

	/// Produce a topological sort, breaking any cycles at the lightest edge.
	std::vector<size_t> topologicalSort() const;

private:

	/// Map from node index to edge lists
	std::map<size_t, Edges> nodes;

	/// Count edges.
	size_t edgeNum;
};

} // end namespace util
} // end namespace OGT_NAMESPACE
#endif /* OGT_UTIL_GRAPH_HPP */
