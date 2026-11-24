// Build on Linux: nvcc -std=c++17 -arch=sm_80 ozaki_ii.cu -lcublas -o ozaki
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thrust/device_vector.h>
#include <vector>

using Wide = __int128;
constexpr int N = 64, bits = 53;  // Square, column-major teaching example.
constexpr int primes[] = {251,241,239,233,229,227,223,211,199,197,193,191,181,179,173,167};
constexpr int blocks = (N*N + 255)/256;
static_assert(N % 4 == 0 && 1LL*N*125*125 <= 2147483647);

void check(int status, const char* operation) {
    if (status) {
        std::fprintf(stderr, "%s failed (status %d)\n", operation, status);
        std::exit(1);
    }
}
#define CHECK(call) check((call), #call)
template<class T> T* ptr(thrust::device_vector<T>& v) {
    return thrust::raw_pointer_cast(v.data());
}

// One thread per row of X or column of Y. Inputs must be finite.
__global__ void prepare(const double* a, bool rows, long long* q, int* scale) {
    int line = blockIdx.x * blockDim.x + threadIdx.x;
    if (line >= N) return;
    int start = rows ? line : line*N, stride = rows ? N : 1;
    double peak = 0;
    for (int j = 0; j < N; ++j) peak = fmax(peak, fabs(a[start+j*stride]));
    int exponent = 0;
    frexp(peak, &exponent);
    scale[line] = exponent - bits;
    for (int j = 0; j < N; ++j) {
        int i = start + j*stride;
        q[i] = (long long)scalbn(a[i], -scale[line]); // truncate toward zero
    }
}

__global__ void residues(const long long* q, int p, signed char* r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N*N) return;
    int v = q[i] % p;
    if (v > p/2) v -= p;
    if (v < -p/2) v += p;
    r[i] = v;
}

// Add (P/p) * ((raw mod p) * inverse mod p), keeping the sum in [0,P).
__global__ void combine(const int* raw, Wide* sum, int p, int inverse, Wide P) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N*N) return;
    int digit = ((raw[i] % p) * inverse) % p;
    if (digit < 0) digit += p;
    Wide s = sum[i] + (P/p)*digit;
    sum[i] = s >= P ? s-P : s;
}

__global__ void restore(const Wide* sum, Wide P, const int* row,
                        const int* col, double* z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N*N) return;
    Wide v = sum[i] > P/2 ? sum[i]-P : sum[i];
    z[i] = scalbn((double)v, row[i%N] + col[i/N]);
}

int main() {
    std::mt19937 rng(594);
    std::uniform_real_distribution<double> uniform(-1,1);
    std::vector<double> x(N*N), y(N*N);
    for (double& a : x) a = uniform(rng);
    for (double& a : y) a = uniform(rng);

    thrust::device_vector<double> X(x.begin(),x.end()), Y(y.begin(),y.end());
    thrust::device_vector<double> Z(N*N), reference(N*N);
    thrust::device_vector<long long> Qx(N*N), Qy(N*N);
    thrust::device_vector<int> row(N), col(N), raw(N*N);
    thrust::device_vector<signed char> A(N*N), B(N*N);
    thrust::device_vector<Wide> sum(N*N, 0);
    cublasHandle_t h;
    CHECK(cublasCreate(&h));

    prepare<<<(N+127)/128,128>>>(ptr(X), true, ptr(Qx), ptr(row));
    prepare<<<(N+127)/128,128>>>(ptr(Y), false, ptr(Qy), ptr(col));
    Wide P = 1;
    for (int p : primes) P *= p;
    if (P/2 < Wide(N) * (Wide(1) << (2*bits))) return 1;

    int one = 1, zero = 0;
    for (int p : primes) {
        int inverse = 1, factor = (P/p) % p;
        while ((factor*inverse) % p != 1) ++inverse;
        residues<<<blocks,256>>>(ptr(Qx), p, ptr(A));
        residues<<<blocks,256>>>(ptr(Qy), p, ptr(B));
        CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N,N,N, &one,
              ptr(A), CUDA_R_8I, N, ptr(B), CUDA_R_8I, N, &zero,
              ptr(raw), CUDA_R_32I, N, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
        combine<<<blocks,256>>>(ptr(raw), ptr(sum), p, inverse, P);
    }
    restore<<<blocks,256>>>(ptr(sum), P, ptr(row), ptr(col), ptr(Z));
    CHECK(cudaGetLastError());
    double alpha = 1, beta = 0;
    CHECK(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N,N,N, &alpha,
                      ptr(X), N, ptr(Y), N, &beta, ptr(reference), N));
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(x.data(), ptr(Z), N*N*sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(y.data(), ptr(reference), N*N*sizeof(double), cudaMemcpyDeviceToHost));
    double error = 0;
    for (int i = 0; i < N*N; ++i) error = std::max(error, std::abs(x[i]-y[i]));
    std::printf("Max absolute difference from cuBLAS FP64: %.3e\n", error);
    CHECK(cublasDestroy(h));
}
