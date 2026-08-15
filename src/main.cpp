#include "Value.h"
#include "Visualizer.h"
#include <cstdlib> 

int main() {
   auto x1 = make_shared<Value>(2.0,"x1");
   auto x2 = make_shared<Value>(0.0,"x2");

   auto w1 = make_shared<Value>(-3.0,"w1");
   auto w2 =make_shared<Value>(-1.0,"w2");

   auto b = make_shared<Value>(6.7,"b");

   auto x1_w1 = x1*w1; x1_w1->label="x1*w1";
   auto x2_w2 = x2*w2; x2_w2->label="x2*w2";

   auto x1_w1_x2_w2 = x1_w1 + x2_w2 ;x1_w1_x2_w2->label="x1*w1+x2*w2";

    auto n = x1_w1_x2_w2 + b; n->label="n";

    auto out = tanh(n); out->label="out";


    draw_dot(out, "graph.dot");
    system("dot -Tpng graph.dot -o graph.png");
    system("open graph.png");
    
    return 0;
}