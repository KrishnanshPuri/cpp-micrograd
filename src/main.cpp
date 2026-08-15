#include "Value.h"
#include "Visualizer.h"
#include "nn.h"
#include <iostream>
#include <vector>

using namespace std;

int main() {
    // 3 inputs -> Layer 1 (4 neurons) -> Layer 2 (4 neurons) -> Output Layer (1 neuron)
    vector<int> nouts = {4, 4, 1};
    MLP n(3, nouts);


    vector<shared_ptr<Value>> x = {
        make_shared<Value>(2.0, "x0"), 
        make_shared<Value>(3.0, "x1"), 
        make_shared<Value>(-1.0, "x2")
    };

    vector<shared_ptr<Value>> out = n(x);
    out[0]->label = "out";

    cout << "Network Output: " << out[0]->data << endl;

   
    vector<shared_ptr<Value>> params = n.parameters();
    cout << "Total Parameters in Network: " << params.size() << endl;

    
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

    
    toposort(out[0]);
    
    
    out[0]->grad = 1.0;

    
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->backward();
    }

   
    draw_dot(out[0], "graph.dot");
    system("dot -Tsvg graph.dot -o graph.svg");
    system("open graph.svg");
    
    return 0;
}