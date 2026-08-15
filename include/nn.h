#pragma once
#include "Value.h"
#include <vector>
#include <cstdlib>

using namespace std;

inline double random_weight() {
    return (rand() / (double)RAND_MAX) * 2.0 - 1.0;
}

class Neuron{
    public:
    vector<shared_ptr<Value>> weights;
    shared_ptr<Value> bias;
    Neuron(int nin,int layer_idx, int neuron_idx){
        string prefix = "L" + to_string(layer_idx) + "_N" + to_string(neuron_idx) + "_";
        for(int i=0;i<nin;i++){
            weights.push_back(make_shared<Value>(random_weight(),prefix + "w"+to_string(i)));
        }
        bias = make_shared<Value>(random_weight(),prefix + "b");
    }

    shared_ptr<Value> operator()(vector<shared_ptr<Value>>x){
        auto act = bias;
        for(int i=0;i<weights.size();i++){
            act = act + weights[i]*x[i];
        }
        return tanh(act);
    }

    vector<shared_ptr<Value>> parameters() {
        vector<shared_ptr<Value>> params = weights;
        params.push_back(bias);
        return params;
    }
};

class Layer {
public:
    vector<Neuron> neurons;

    Layer(int nin, int nout,int layer_idx) {
        for (int i = 0; i < nout; ++i) {
            neurons.push_back(Neuron(nin,layer_idx,i));
        }
    }

   
    vector<shared_ptr<Value>> operator()(const vector<shared_ptr<Value>>& x) {
        vector<shared_ptr<Value>> outs;
        for (auto& n : neurons) {
            outs.push_back(n(x));
        }
        return outs;
    }

    vector<shared_ptr<Value>> parameters() {
        vector<shared_ptr<Value>> params;
        for (auto& n : neurons) {
            auto n_params = n.parameters();
            params.insert(params.end(), n_params.begin(), n_params.end());
        }
        return params;
    }
};

class MLP{
    public:
    vector<Layer>layers;
    MLP(int nin,vector<int>nouts){
        int sz = nin;
        for(int i=0;i<nouts.size();i++){
            layers.push_back(Layer(sz,nouts[i],i));
            sz = nouts[i];
        }
    }

    vector<shared_ptr<Value>> operator()(vector<shared_ptr<Value>> x) {
        for (auto& layer : layers) {
            x = layer(x); 
        }
        return x;
        }


vector<shared_ptr<Value>> parameters() {
        vector<shared_ptr<Value>> params;
        for (auto& layer : layers) {
            auto l_params = layer.parameters();
            params.insert(params.end(), l_params.begin(), l_params.end());
        }
        return params;
    }


};




