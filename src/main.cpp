#include "Value.h"
#include "Visualizer.h"
#include "nn.h"
#include <iostream>
#include <vector>
#include <functional>
#include <unordered_set>

using namespace std;

int main() {
    MLP n(3, {4, 4, 1});

    vector<vector<shared_ptr<Value>>> xs = {
        {make_shared<Value>(2.0, "x0"), make_shared<Value>(3.0, "x1"), make_shared<Value>(-1.0, "x2")},
        {make_shared<Value>(3.0, "x0"), make_shared<Value>(-1.0, "x1"), make_shared<Value>(0.5, "x2")},
        {make_shared<Value>(0.5, "x0"), make_shared<Value>(1.0, "x1"), make_shared<Value>(1.0, "x2")},
        {make_shared<Value>(1.0, "x0"), make_shared<Value>(1.0, "x1"), make_shared<Value>(-1.0, "x2")}
    };
    
    vector<double> ys = {1.0, -1.0, -1.0, 1.0};

    for (int k = 0; k < 20; k++) {
        auto loss = make_shared<Value>(0.0);
        
        for (size_t i = 0; i < 4; i++) {
            auto ypred = n(xs[i])[0];
            auto target = make_shared<Value>(-ys[i]);
            auto diff = ypred + target;
            loss = loss + (diff * diff);
        }

        for (auto& p : n.parameters()) {
            p->grad = 0.0;
        }

        vector<shared_ptr<Value>> topo;
        unordered_set<shared_ptr<Value>> vis;
        
        function<void(shared_ptr<Value>)> toposort;
        toposort = [&](shared_ptr<Value> node) {
            if (!node || vis.count(node)) return;
            vis.insert(node);
            for (auto& child : node->prev) {
                toposort(child);
            }
            topo.push_back(node);
        };

        toposort(loss);
        
        loss->grad = 1.0;
        
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->backward();
        }

        double alpha = 0.05;
        for (auto& p : n.parameters()) {
            p->data += -alpha * p->grad;
        }

        cout << "Epoch " << k << " | Loss: " << loss->data << endl;
    }

    cout << "\n--- Final Predictions ---" << endl;
    for (size_t i = 0; i < 4; i++) {
        cout << "Target: " << ys[i] << " | Prediction: " << n(xs[i])[0]->data << endl;
    }
    
   auto final_out = n(xs[0])[0];
    final_out->label = "final_out";
    draw_dot(final_out, "graph.dot");
    system("dot -Tsvg graph.dot -o graph.svg");
    system("open graph.svg");
    return 0;
}