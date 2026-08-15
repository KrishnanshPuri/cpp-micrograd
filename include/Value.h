#pragma once
#include <iostream>
#include <unordered_set>
#include <memory>
#include <string>
#include<cmath>
#include<algorithm>
#include<functional>
using namespace std;

class Value {
public:
    double data;
    unordered_set<shared_ptr<Value>> prev;
    double grad=0.0;
    char op;
    function<void()> backward;
    string label;
    Value(double data,string label = "") : data(data), op(' '),label(label) {
        backward= [](){};
    }

    Value(double data, unordered_set<shared_ptr<Value>> prev, char op,string label = "") 
        : data(data), prev(prev), op(op),label(label) {
             backward= [](){};
        }
};


inline shared_ptr<Value> operator+(shared_ptr<Value> a, shared_ptr<Value> b) {
    unordered_set<shared_ptr<Value>> children = {a, b};
     auto y = make_shared<Value>(a->data + b->data, children, '+');
     Value* y_ptr = y.get();
     auto f = [a,b,y_ptr](){
      a->grad=y_ptr->grad;
      b->grad= y_ptr->grad;
    };
   y->backward=f;
    return y;
}

inline shared_ptr<Value> operator*(shared_ptr<Value> a, shared_ptr<Value> b) {
    unordered_set<shared_ptr<Value>> children = {a, b};
    auto y = make_shared<Value>(a->data * b->data, children, '*');
    Value* y_ptr = y.get();
    auto f = [a,b,y_ptr](){
      a->grad=y_ptr->grad * b->data;
      b->grad= y_ptr->grad* a->data;
    };
   y->backward=f;
   return y;
}

inline shared_ptr<Value> tanh(shared_ptr<Value> x){
    double y = tanh(x->data);
    unordered_set<shared_ptr<Value>> children = {x};
    return make_shared<Value>(y, children, 't');
}

inline shared_ptr<Value> relu(shared_ptr<Value> x) {
    double out = max(0.0, x->data);
    unordered_set<shared_ptr<Value>> children = {x};
    return make_shared<Value>(out, children, 'r');
}