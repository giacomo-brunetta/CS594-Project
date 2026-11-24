// Build on Linux: nvcc -std=c++17 -arch=sm_80 ozaki_ii.cu -lcublas -o ozaki
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thrust/device_vector.h>
#include <vector>

using Wide = __int128;
// Run: ./ozaki [M K N] for X(M,K) times Y(K,N).
constexpr int primes[] = {251,241,239,233,229,227,223,211,199,197,193,191,181,179,173,167};
int blocks(int count) { return 1 + (count-1)/256; }

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
__global__ void prepare(const double* a, int lines, int length, int ld, bool rows, int bits,
                        long long* q, int* scale) {
    int line = blockIdx.x * blockDim.x + threadIdx.x;
    if (line >= lines) return;
    int start = rows ? line : line*ld, stride = rows ? ld : 1;
    double peak = 0;
    for (int j = 0; j < length; ++j) peak = fmax(peak, fabs(a[start+j*stride]));
    int exponent = 0;
    frexp(peak, &exponent);
    scale[line] = exponent - bits;
    for (int j = 0; j < length; ++j) {
        int i = start + j*stride;
        q[i] = (long long)scalbn(a[i], -scale[line]); // truncate toward zero
    }
}

__global__ void residues(const long long* q, int count, int p, signed char* r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int v = q[i] % p;
    if (v > p/2) v -= p;
    if (v < -p/2) v += p;
    r[i] = v;
}

// Add (P/p) * ((raw mod p) * inverse mod p), keeping the sum in [0,P).
__global__ void combine(const int* raw, Wide* sum, int count, int p, int inverse, Wide P) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int digit = ((raw[i] % p) * inverse) % p;
    if (digit < 0) digit += p;
    Wide s = sum[i] + (P/p)*digit;
    sum[i] = s >= P ? s-P : s;
}

__global__ void restore(const Wide* sum, Wide P, const int* row,
                        const int* col, int count, int ld, double* z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    Wide v = sum[i] > P/2 ? sum[i]-P : sum[i];
    z[i] = scalbn((double)v, row[i%ld] + col[i/ld]);
}

int dimension(const char* arg) {
    char* end;
    long value = std::strtol(arg, &end, 10);
    if (*end || value < 1 || value > INT_MAX-3) {
        std::fprintf(stderr, "Invalid dimension: %s\n", arg);
        std::exit(1);
    }
    return int(value);
}

int main(int argc, char** argv) {
    if (argc != 1 && argc != 4) {
        std::fprintf(stderr, "Usage: %s [M K N]\n", argv[0]);
        return 1;
    }
    int m = 64, k = 64, n = 64;
    if (argc == 4) {
        m = dimension(argv[1]);
        k = dimension(argv[2]);
        n = dimension(argv[3]);
    }
    // Zero padding gives the INT8 operands aligned columns and inner dimension.
    int ld = (m+3)/4*4, inner = (k+3)/4*4;
    if (ld > INT_MAX/inner || inner > INT_MAX/n || ld > INT_MAX/n) {
        std::fprintf(stderr, "Padded matrices exceed the 32-bit index range.\n");
        return 1;
    }
    int nx = ld*inner, ny = inner*n, nz = ld*n;
    std::mt19937 rng(594);
    std::uniform_real_distribution<double> uniform(-1,1);
    std::vector<double> x(nx), y(ny);
    for (int j = 0; j < k; ++j)
        for (int i = 0; i < m; ++i) x[i+j*ld] = uniform(rng);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < k; ++i) y[i+j*inner] = uniform(rng);

    thrust::device_vector<double> X(x.begin(),x.end()), Y(y.begin(),y.end());
    thrust::device_vector<double> Z(nz), reference(nz);
    thrust::device_vector<long long> Qx(nx), Qy(ny);
    thrust::device_vector<int> row(ld), col(n), raw(nz);
    thrust::device_vector<signed char> A(nx), B(ny);
    thrust::device_vector<Wide> sum(nz, 0);
    cublasHandle_t h;
    CHECK(cublasCreate(&h));

    Wide P = 1;
    for (int p : primes) P *= p;
    int bits = 53;
    while ((Wide(1) << (2*bits)) > P/2/k) --bits;
    prepare<<<blocks(ld),256>>>(ptr(X), ld, inner, ld, true, bits, ptr(Qx), ptr(row));
    prepare<<<blocks(n),256>>>(ptr(Y), n, inner, inner, false, bits, ptr(Qy), ptr(col));

    int one = 1, zero = 0;
    for (int p : primes) {
        int inverse = 1, factor = (P/p) % p;
        while ((factor*inverse) % p != 1) ++inverse;
        residues<<<blocks(nx),256>>>(ptr(Qx), nx, p, ptr(A));
        residues<<<blocks(ny),256>>>(ptr(Qy), ny, p, ptr(B));
        // A partial dot product fits int32 since |residue| <= 125.
        constexpr int chunk = (INT_MAX/(125*125))/4*4;
        for (int first = 0; first < inner; ) {
            int width = std::min(chunk, inner-first);
            CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, ld,n,width, &one,
                  ptr(A)+first*ld, CUDA_R_8I, ld, ptr(B)+first, CUDA_R_8I, inner,
                  &zero, ptr(raw), CUDA_R_32I, ld, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
            combine<<<blocks(nz),256>>>(ptr(raw), ptr(sum), nz, p, inverse, P);
            first += width;
        }
    }
    restore<<<blocks(nz),256>>>(ptr(sum), P, ptr(row), ptr(col), nz, ld, ptr(Z));
    CHECK(cudaGetLastError());
    double alpha = 1, beta = 0;
    CHECK(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m,n,k, &alpha,
                      ptr(X), ld, ptr(Y), inner, &beta, ptr(reference), ld));
    CHECK(cudaDeviceSynchronize());
    x.resize(nz);
    y.resize(nz);
    CHECK(cudaMemcpy(x.data(), ptr(Z), nz*sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(y.data(), ptr(reference), nz*sizeof(double), cudaMemcpyDeviceToHost));
    double error = 0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            error = std::max(error, std::abs(x[i+j*ld]-y[i+j*ld]));
    std::printf("Max absolute difference from cuBLAS FP64: %.3e\n", error);
    CHECK(cublasDestroy(h));
}
