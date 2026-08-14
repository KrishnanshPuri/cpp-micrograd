#include "Value.h"
#include "Visualizer.h"
#include <cstdlib> 

int main() {
    auto a = make_shared<Value>(-3.0,"a");
    auto b = make_shared<Value>(2.0,"b");
    auto c = make_shared<Value>(10.0,"c");

    auto e = a * b;
    e->label = "e";

    auto d = e + c;
    d->label = "d";

    draw_dot(d, "graph.dot");
    
   
    system("dot -Tpng graph.dot -o graph.png");
    system("open graph.png");
    
    return 0;
}