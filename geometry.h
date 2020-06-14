#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

#include <cmath>
#include <vector>
#include <cassert>
#include <iostream>

template <size_t DIM, typename T> struct vec {
    __host__ __device__ vec() { for (size_t i=DIM; i--; data_[i] = T()); }
    __host__ __device__      T& operator[](const size_t i)       { assert(i<DIM); return data_[i]; }
    __host__ __device__ const T& operator[](const size_t i) const { assert(i<DIM); return data_[i]; }
private:
    T data_[DIM];
};

typedef vec<2, float> Vec2f;
typedef vec<3, float> Vec3f;
typedef vec<3, int  > Vec3i;
typedef vec<4, float> Vec4f;

template <typename T> struct vec<2,T> {
    __host__ __device__ vec() : x(T()), y(T()) {}
    __host__ __device__ vec(T X, T Y) : x(X), y(Y) {}
    __host__ __device__       T& operator[](const size_t i)       { assert(i<2); return i<=0 ? x : y; }
    __host__ __device__ const T& operator[](const size_t i) const { assert(i<2); return i<=0 ? x : y; }
    T x,y;
};

template <typename T> struct vec<3,T> {
    __host__ __device__ vec() : x(T()), y(T()), z(T()) {}
    __host__ __device__ vec(T X, T Y, T Z) : x(X), y(Y), z(Z) {}
    __host__ __device__       T& operator[](const size_t i)       { assert(i<3); return i<=0 ? x : (1==i ? y : z); }
    __host__ __device__ const T& operator[](const size_t i) const { assert(i<3); return i<=0 ? x : (1==i ? y : z); }
    __host__ __device__ float norm() { return std::sqrt(x*x+y*y+z*z); }
    __host__ __device__ vec<3,T>  normalize(T l=1) { float t = (T)1.0 / norm(); return vec<3,T>(x*t, y*t,z*t); }
    T x,y,z;
};



template <typename T> struct vec<4,T> {
    __host__ __device__ vec() : x(T()), y(T()), z(T()), w(T()) {}
    __host__ __device__ vec(T X, T Y, T Z, T W) : x(X), y(Y), z(Z), w(W) {}
    __host__ __device__       T& operator[](const size_t i)       { assert(i<4); return i<=0 ? x : (1==i ? y : (2==i ? z : w)); }
    __host__ __device__ const T& operator[](const size_t i) const { assert(i<4); return i<=0 ? x : (1==i ? y : (2==i ? z : w)); }
    T x,y,z,w;
};


template<size_t DIM,typename T>
__host__ __device__ T operator*(const vec<DIM,T>& lhs, const vec<DIM,T>& rhs) {
    T ret = T();
    for (size_t i=DIM; i--; ret+=lhs[i]*rhs[i]);
    return ret;
}

template<size_t DIM,typename T>
__host__ __device__ vec<DIM,T> operator+(vec<DIM,T> lhs, const vec<DIM,T>& rhs) {
    for (size_t i=DIM; i--; lhs[i]+=rhs[i]);
    return lhs;
}

template<size_t DIM,typename T>
__host__ __device__ vec<DIM,T> operator-(vec<DIM,T> lhs, const vec<DIM,T>& rhs) {
    for (size_t i=DIM; i--; lhs[i]-=rhs[i]);
    return lhs;
}

template<size_t DIM,typename T,typename U> vec<DIM,T> operator*(const vec<DIM,T> &lhs, const U& rhs) {
    vec<DIM,T> ret;
    for (size_t i=DIM; i--; ret[i]=lhs[i]*rhs);
    return ret;
}

template<size_t DIM,typename T>
__host__ __device__ vec<DIM,T> operator-(const vec<DIM,T> &lhs) {
    return lhs*T(-1);
}

template <typename T>
__host__ __device__ vec<3,T> cross(vec<3,T> v1, vec<3,T> v2) {
    return vec<3,T>(v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x);
}

__host__ __device__ Vec3f operator*(const Vec3f &v, float t) {
    return Vec3f(t*v.x, t*v.y, t*v.z);
}

__host__ __device__ Vec3f operator*(float t, const Vec3f &v) {
    return Vec3f(t*v.x, t*v.y, t*v.z);
}

//__host__ __device__ Vec3f unit_vector(Vec3f v) {
//    float t =  1.0/v.norm();
//    return Vec3f(t*v.x, t*v.y, t*v.z);
//
//}

__host__ __device__ inline float dot(const Vec3f &v1, const Vec3f &v2) {
    return v1.x *v2.x + v1.y *v2.y  + v1.z *v2.z;
}

__host__ __device__ inline Vec3f cross(const Vec3f &v1, const Vec3f &v2) {
    return Vec3f( (v1.y*v2.z - v1.z*v2.y),
                 (-(v1.x*v2.z - v1.z*v2.x)),
                 (v1.x*v2.y - v1.y*v2.x));
}

__host__ __device__ inline Vec3f element_dot(const Vec3f &v1, const Vec3f &v2) {
    return Vec3f (v1.x *v2.x, v1.y *v2.y, v1.z *v2.z);
}


template <size_t DIM, typename T> std::ostream& operator<<(std::ostream& out, const vec<DIM,T>& v) {
    for(unsigned int i=0; i<DIM; i++) out << v[i] << " " ;
    return out ;
}
#endif //__GEOMETRY_H__

