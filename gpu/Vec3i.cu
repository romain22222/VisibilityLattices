#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#ifndef VEC3I_H
#define VEC3I_H

class Vec3i {
public:
  int x, y, z;

  CUDA_HOSTDEV Vec3i() : x(0), y(0), z(0) {}

  CUDA_HOSTDEV Vec3i(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}

  CUDA_HOSTDEV Vec3i operator+(const Vec3i &other) const {
    return {x + other.x, y + other.y, z + other.z};
  }

  CUDA_HOSTDEV Vec3i operator-(const Vec3i &other) const {
    return {x - other.x, y - other.y, z - other.z};
  }

  CUDA_HOSTDEV Vec3i operator*(int val) const {
    return {x * val, y * val, z * val};
  }

  CUDA_HOSTDEV Vec3i operator/(int val) const {
    return {x / val, y / val, z / val};
  }

  CUDA_HOSTDEV bool operator==(const Vec3i &other) const {
    return x == other.x && y == other.y && z == other.z;
  }

  CUDA_HOSTDEV bool operator!=(const Vec3i &other) const {
    return !(*this == other);
  }

  CUDA_HOSTDEV int &operator[](int index) {
    return (index == 0) ? x : (index == 1) ? y : z;
  }

  CUDA_HOSTDEV const int &operator[](int index) const {
    return (index == 0) ? x : (index == 1) ? y : z;
  }

  CUDA_HOSTDEV Vec3i &operator+=(const Vec3i &other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
  }
};

#endif // VEC3I_H