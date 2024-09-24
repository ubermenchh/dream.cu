#ifndef VECTOR_H
#define VECTOR_H 

#include <cmath> 
#include <iostream> 

class Vector {
    public:
        double e[3];

        __host__ __device__ Vector() {}
        __host__ __device__ Vector(double e0, double e1, double e2) {
            e[0] = e0; e[1] = e1; e[2] = e2;
        }

        __host__ __device__ double x() const {return e[0];}
        __host__ __device__ double y() const {return e[1];}
        __host__ __device__ double z() const {return e[2];}

        __host__ __device__ Vector operator-() const {return Vector(-e[0], -e[1], -e[2]);}
        __host__ __device__ double operator[](int i) const {return e[i];}
        __host__ __device__ double& operator[](int i) {return e[i];}

        __host__ __device__ inline Vector& operator+=(const Vector& v2);
        __host__ __device__ inline Vector& operator-=(const Vector& v2);
        __host__ __device__ inline Vector& operator*=(const Vector& v2);
        __host__ __device__ inline Vector& operator/=(const Vector& v2);
        __host__ __device__ inline Vector& operator*=(const double t);
        __host__ __device__ inline Vector& operator/=(const double t);

        __host__ __device__ double length() const {
            return std::sqrt(length_squared());
        }
        __host__ __device__ double length_squared() const {
            return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
        }
};

// Vector Utility Functions 
__host__ __device__ inline std::ostream& operator<<(std::ostream& out, const Vector& v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline Vector operator+(const Vector& u, const Vector& v) {
    return Vector(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline Vector operator-(const Vector& u, const Vector& v) {
    return Vector(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline Vector operator*(const Vector& u, const Vector& v) {
    return Vector(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline Vector operator*(double t, const Vector& v) {
    return Vector(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline Vector operator*(const Vector& v, double t) {
    return t * v;
}

__host__ __device__ inline Vector operator/(const Vector& v, double t) {
    return (1 / t) * v;
}


__host__ __device__ inline double dot(const Vector& u, const Vector& v) {
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

__host__ __device__ inline Vector cross(const Vector& u, const Vector& v) {
    return Vector(u.e[0] * v.e[2] - u.e[2] * v.e[1],
                  u.e[2] * v.e[0] - u.e[0] * v.e[2],
                  u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline Vector unit_vector(const Vector& v) {
    return v / v.length();
}

__host__ __device__ inline Vector& Vector::operator+=(const Vector& v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}
    
__host__ __device__ inline Vector& Vector::operator*=(const Vector& v) {
    e[0] *= v.e[0];
    e[1] *= v.e[1];
    e[2] *= v.e[2];
    return *this;
}

__host__ __device__ inline Vector& Vector::operator/=(const Vector& v) {
    e[0] /= v.e[0];
    e[1] /= v.e[1];
    e[2] /= v.e[2];
    return *this;
}
    
__host__ __device__ inline Vector& Vector::operator-=(const Vector& v) {
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
}

__host__ __device__ inline Vector& Vector::operator*=(double t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}    
    
__host__ __device__ inline Vector& Vector::operator/=(double t) {
    double k = 1.0 / t;
    
    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
    return *this;
}  

#endif // VECTOR_H
