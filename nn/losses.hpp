#ifndef LOSSES_HPP
#define LOSSES_HPP

#include"tensor.hpp"
#include<cmath>

namespace LibCN{
    namespace Losses{
        template<Element T>T MSE(const Tensor<T>&x,const Tensor<T>&e){
            T res=T();
            if(x.getShape()==e.getShape()){
                T sum=T(0);
                for(size_t i=0;i<x.getShape()[0];i++)for(size_t j=0;j<x.getShape()[1];j++)sum+=(x(i,j)-e(i,j))*(x(i,j)-e(i,j));
                res=sum/T(2);
            }
            return res;
        }

        template<Element T>Tensor<T>MSE_d(const Tensor<T>&x,const Tensor<T>&e){
            return x-e;
        }

        template<Element T>T MAE(const Tensor<T>&x,const Tensor<T>&e){
            T res=T();
            if(x.getShape()==e.getShape()){
                T sum=T(0);
                for(size_t i=0;i<x.getShape()[0];i++)for(size_t j=0;j<x.getShape()[1];j++){
                    T tmp=(x(i,j)-e(i,j));
                    if(tmp>T(0))sum+=tmp;
                    else sum-=tmp;
                }
                res=sum/(x.h*x.l);
            }
            return res;
        }

        template<Element T>Tensor<T>MAE_d(const Tensor<T>&x,const Tensor<T>&e){
            Tensor<T>res;
            if(x.getShape()==e.getShape()){
                res.resize(x.getDimension(),x.getShape());
                for(size_t i=0;i<x.getShape()[0];i++)for(size_t j=0;j<x.getShape()[1];j++)res(i,j)=x(i,j)>=e(i,j)?T(1):T(-1);
            }
            return res;
        }

        template<Element T>T cross_entropy(const Tensor<T>&x,const Tensor<T>&e){
            T res=T();
            if(x.getShape()==e.getShape()){
                constexpr T eps=static_cast<T>(1e-12);
                for(size_t i=0;i<x.getShape()[0];++i)for(size_t j=0;j<x.getShape()[1];++j){
                    T v=x(i,j);
                    if(v<eps)v=eps;
                    if(v>static_cast<T>(1)-eps)v=static_cast<T>(1)-eps;
                    res-=e(i,j)*std::log(v);
                }
            }
            return res;
        }

        template<Element T>Tensor<T>cross_entropy_d(const Tensor<T>&x,const Tensor<T>&e){
            Tensor<T>res(x.getDimension(),x.getShape());
            if(x.getShape()!=e.getShape())return res;
            constexpr T eps=static_cast<T>(1e-12);
            for(size_t i=0;i<x.getShape()[0];++i)for(size_t j=0;j<x.getShape()[1];++j){
                T v=x(i,j);
                if(v<eps)v=eps;
                if(v>static_cast<T>(1)-eps)v=static_cast<T>(1)-eps;
                res(i,j)=T(-1)*e(i,j)/v;
            }
            return res;
        }
    }
}

#endif