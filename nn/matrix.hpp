#ifndef MATRIX_HPP
#define MATRIX_HPP

#include<vector>
#include<iostream>
#include<functional>
#include<concepts>

namespace LibCN{
    template<typename T>concept Element=requires(T a,T b,std::iostream&os){
        {a+b}->std::same_as<T>;
        {a-b}->std::same_as<T>;
        {a*b}->std::same_as<T>;
        {os<<a}->std::same_as<std::ostream&>;
        {a>b}->std::same_as<bool>;
        {a<b}->std::same_as<bool>;
        {a>=b}->std::same_as<bool>;
        {a<=b}->std::same_as<bool>;
        {a==b}->std::same_as<bool>;
        {a!=b}->std::same_as<bool>;
    };

    enum class Direction{
        Up,
        Down,
        Left,
        Right
    };

    template<Element T>struct Matrix{
        std::vector<std::vector<T>>mv;
        size_t h,l;

        Matrix(){
            mv.resize(0);
            h=0;
            l=0;
        }

        Matrix(size_t h,size_t l){
            this->h=h;
            this->l=l;
            mv.resize(h);
            for(size_t i=0;i<h;i++)mv[i].resize(l);
        }

        Matrix(const std::vector<std::vector<T>>&a){
            this->mv=a;
            this->h=a.size();
            this->l=a[0].size();
        }

        Matrix(const Matrix<T>&a){
            this->h=a.h;
            this->l=a.l;
            this->mv=a.mv;
        }

        Matrix<T>&operator=(const Matrix<T>&a){
            this->h=a.h;
            this->l=a.l;
            this->mv=a.mv;
            return*this;
        }

        Matrix(std::initializer_list<std::initializer_list<T>>init){
            h=init.size();
            l=init.begin()->size();
            mv.reserve(h);
            for(const auto&row:init){
                mv.emplace_back(row);
            }
        }

        std::vector<T>&operator[](size_t index){
            return mv[index];
        }

        const std::vector<T>&operator[](size_t index)const{
            return mv[index];
        }

        friend std::ostream&operator<<(std::ostream&os,const Matrix<T>&a){
            if(a.h==0)os<<"{ NULL }";
            for(size_t i=0;i<a.h;i++){
                if(i==0)os<<"{";
                else os<<" ";
                os<<" ";
                for(size_t j=0;j<a.l;j++){
                    os<<a[i][j]<<" ";
                }
                if(i==a.h-1)os<<"}";
                else os<<"\n";
            }
            return os;
        }

        void resize(size_t h,size_t l){
            this->h=h;
            this->l=l;
            mv.resize(h);
            for(size_t i=0;i<h;i++){
                mv[i].resize(l);
            }
        }

        Matrix<T>append(const Matrix<T>&a,Direction d)const{
            Matrix<T>res;
            switch(d){
                case Direction::Up:{
                    if(this->l!=a.l)break;
                    res.resize(this->h+a.h,this->l);
                    for(size_t i=0;i<a.h;i++)for(size_t j=0;j<this->l;j++)res[i][j]=a[i][j];
                    for(size_t i=0;i<this->h;i++)for(size_t j=0;j<this->l;j++)res[i+a.h][j]=this->mv[i][j];
                }break;
                case Direction::Down:{
                    if(this->l!=a.l)break;
                    res.resize(this->h+a.h,this->l);
                    for(size_t i=0;i<this->h;i++)for(size_t j=0;j<this->l;j++)res[i][j]=this->mv[i][j];
                    for(size_t i=0;i<a.h;i++)for(size_t j=0;j<this->l;j++)res[i+this->h][j]=a[i][j];
                }break;
                case Direction::Left:{
                    if(this->h!=a.h)break;
                    res.resize(this->h,this->l+a.l);
                    for(size_t i=0;i<this->h;i++)for(size_t j=0;j<a.l;j++)res[i][j]=a[i][j];
                    for(size_t i=0;i<this->h;i++)for(size_t j=0;j<this->l;j++)res[i][j+a.l]=this->mv[i][j];
                }break;
                case Direction::Right:{
                    if(this->h!=a.h)break;
                    res.resize(this->h,this->l+a.l);
                    for(size_t i=0;i<this->h;i++)for(size_t j=0;j<this->l;j++)res[i][j]=this->mv[i][j];
                    for(size_t i=0;i<this->h;i++)for(size_t j=0;j<a.l;j++)res[i][j+this->l]=a[i][j];
                }break;
            }
            return res;
        }

        Matrix<T>transpose()const{
            Matrix<T>res(l,h);
            for(size_t i=0;i<h;i++)for(size_t j=0;j<l;j++)res[j][i]=this->mv[i][j];
            return res;
        }

        Matrix<T>operator+(const Matrix<T>&a)const{
            Matrix<T>res;
            if(this->h==a.h&&this->l==a.l){
                res.resize(a.h,a.l);
                for(size_t i=0;i<a.h;i++)for(size_t j=0;j<a.l;j++)res[i][j]=a[i][j]+this->mv[i][j];
            }
            return res;
        }

        Matrix<T>&operator+=(const Matrix<T>&a){
            *this=*this+a;
            return*this;
        }

        Matrix<T>operator-(const Matrix<T>&a)const{
            Matrix<T>res;
            if(this->h==a.h&&this->l==a.l){
                res.resize(a.h,a.l);
                for(size_t i=0;i<a.h;i++)for(size_t j=0;j<a.l;j++)res[i][j]=this->mv[i][j]-a[i][j];
            }
            return res;
        }

        Matrix<T>&operator-=(const Matrix<T>&a){
            *this=*this-a;
            return*this;
        }

        Matrix<T>operator*(const T&a)const{
            Matrix<T>res(h,l);
            for(size_t i=0;i<h;i++)for(size_t j=0;j<l;j++)res[i][j]=mv[i][j]*a;
            return res;
        }

        friend Matrix<T>operator*(const T&a,const Matrix<T>&b){
            return b*a;
        }

        Matrix<T>&operator*=(const T&a){
            *this=*this*a;
            return*this;
        }

        Matrix<T>operator*(const Matrix<T>&a)const{
            Matrix<T>res;
            if(this->l==a.h){
                res.resize(this->h,a.l);
                for(size_t i=0;i<res.h;i++)for(size_t j=0;j<res.l;j++){
                    res[i][j]=T{};
                    for(size_t k=0;k<this->l;k++)res[i][j]+=(this->mv[i][k]*a[k][j]);
                }
            }
            return res;
        }

        Matrix<T>&operator*=(const Matrix&a){
            *this=*this*a;
            return*this;
        }

        Matrix<T>hadamard(const Matrix<T>&a)const{
            Matrix<T>res;
            if(this->h==a.h&&this->l==a.l){
                res.resize(a.h,a.l);
                for(size_t i=0;i<a.h;i++)for(size_t j=0;j<a.l;j++)res[i][j]=this->mv[i][j]*a[i][j];
            }
            return res;
        }

        Matrix<T>apply(const std::function<T(T)>&a)const{
            Matrix<T>res(h,l);
            for(size_t i=0;i<h;i++)for(size_t j=0;j<l;j++)res[i][j]=a(mv[i][j]);
            return res;
        }
    };
}

#endif