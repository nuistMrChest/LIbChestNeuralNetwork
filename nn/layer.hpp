#ifndef LAYER_HPP
#define LAYER_HPP

#include"matrix.hpp"
#include<functional>

namespace LibCN{
    template<Element T>struct Layer{
        std::function<T(T)>activation;
        size_t in_size;
        size_t out_size;
        Matrix<T>W;
        Matrix<T>b;
        Matrix<T>last_input;
        Matrix<T>z;

        Layer(){
            in_size=0;
            out_size=0;
            W=Matrix<T>();
            b=Matrix<T>();
            last_input=Matrix<T>();
            z=Matrix<T>();
        }

        Layer(size_t i,size_t o){
            in_size=i;
            out_size=o;
            W.resize(o,i);
            b.resize(o,1);
            last_input.resize(i,1);
            z.resize(o,1);
        }

        Matrix<T>forward(const Matrix<T>&input)const{
            Matrix<T>res(in_size,1);
            if(input.h==in_size&&input.l==1)res=((W*input)+b).apply(activation);
            return res;
        }

        
    };
}

#endif