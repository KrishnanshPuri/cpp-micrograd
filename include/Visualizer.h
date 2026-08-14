#pragma once
#include "Value.h"
#include <fstream>
#include <unordered_set>
#include <vector>
#include <string>

using namespace std;

inline void draw_dot(const shared_ptr<Value>& root, const string& filename = "graph.dot") {
    unordered_set<shared_ptr<Value>> nodes;
    vector<pair<shared_ptr<Value>, shared_ptr<Value>>> edges;

    auto dfs = [&](auto& self, const shared_ptr<Value>& node) -> void {
        if (!node || nodes.count(node)) return;
        nodes.insert(node);
        for (const auto& child : node->prev) {
            edges.push_back({child, node});
            self(self, child);
        }
    };
    
    dfs(dfs, root);

    ofstream out(filename);
    out << "digraph G {\n";
    out << "    rankdir=LR;\n";
    out << "    node [fontsize=10];\n";

    for (const auto& n : nodes) {
        uintptr_t id = reinterpret_cast<uintptr_t>(n.get());
        
        out << "    node_" << id << " [label=\"data " << n->data << "\", shape=box];\n";

        if (n->op != ' ') {
            out << "    op_" << id << " [label=\"" << n->op << "\", shape=circle];\n";
            out << "    op_" << id << " -> node_" << id << ";\n";
        }
    }

    for (const auto& edge : edges) {
        uintptr_t child_id = reinterpret_cast<uintptr_t>(edge.first.get());
        uintptr_t parent_id = reinterpret_cast<uintptr_t>(edge.second.get());

        if (edge.second->op != ' ') {
            out << "    node_" << child_id << " -> op_" << parent_id << ";\n";
        } else {
            out << "    node_" << child_id << " -> node_" << parent_id << ";\n";
        }
    }

    out << "}\n";
    out.close();
}