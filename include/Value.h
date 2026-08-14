#pragma once
#include <iostream>
#include <unordered_set>
#include <memory>

using namespace std;

class Value {
public:
    double data;
    unordered_set<shared_ptr<Value>> prev;
    char op;

    Value(double data) : data(data), op(' ') {}

    Value(double data, unordered_set<shared_ptr<Value>> prev, char op) 
        : data(data), prev(prev), op(op) {}
};


inline shared_ptr<Value> operator+(shared_ptr<Value> a, shared_ptr<Value> b) {
    unordered_set<shared_ptr<Value>> children = {a, b};
    return make_shared<Value>(a->data + b->data, children, '+');
}

inline shared_ptr<Value> operator*(shared_ptr<Value> a, shared_ptr<Value> b) {
    unordered_set<shared_ptr<Value>> children = {a, b};
    return make_shared<Value>(a->data * b->data, children, '*');
}