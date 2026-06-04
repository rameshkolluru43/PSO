#include <cuda_runtime.h>
#include <math_constants.h>
#include <math_functions.h>

struct Particle2D {
    float2 position;
    float2 velocity;
    float2 best_position;
    float fitness;
    float best_fitness;
};

struct Params2D {
    float w;
    float c1;
    float c2;
    unsigned int count;
    float2 gbest;
    float xmin;
    float xmax;
    float vmin;
    float vmax;
    unsigned int objective_type;
    unsigned int cluster_count;
    unsigned int use_cluster_best;
    unsigned int padding;
};

struct KMeansParams2D {
    unsigned int particle_count;
    unsigned int cluster_count;
};

struct ParamsND {
    float w;
    float c1;
    float c2;
    unsigned int count;
    unsigned int dim;
    unsigned int objective_type;
    unsigned int cluster_count;
    unsigned int use_cluster_best;
    float xmin;
    float xmax;
    float vmin;
    float vmax;
};

struct KMeansParamsND {
    unsigned int particle_count;
    unsigned int cluster_count;
    unsigned int dim;
};

__device__ inline float clampf(float x, float lo, float hi)
{
    return fminf(fmaxf(x, lo), hi);
}

__device__ inline float2 add2(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ inline float2 sub2(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ inline float2 mul2(float s, float2 a)
{
    return make_float2(s * a.x, s * a.y);
}

__device__ inline float dot2(float2 a, float2 b)
{
    return a.x * b.x + a.y * b.y;
}

__device__ inline float sphere2d(float2 x)
{
    return dot2(x, x);
}

__device__ inline float rastrigin2d(float2 x)
{
    const float a = 10.0f;
    const float twopi = 6.283185307179586f;
    return 2.0f * a + (x.x * x.x - a * cosf(twopi * x.x)) + (x.y * x.y - a * cosf(twopi * x.y));
}

__device__ inline float ackley2d(float2 x)
{
    const float a = 20.0f;
    const float b = 0.2f;
    const float c = 6.283185307179586f;
    float sq = 0.5f * (x.x * x.x + x.y * x.y);
    float cs = 0.5f * (cosf(c * x.x) + cosf(c * x.y));
    return -a * expf(-b * sqrtf(sq)) - expf(cs) + a + 2.718281828459045f;
}

__device__ inline float himmelblau2d(float2 x)
{
    float t1 = x.x * x.x + x.y - 11.0f;
    float t2 = x.x + x.y * x.y - 7.0f;
    return t1 * t1 + t2 * t2;
}

__device__ inline float six_hump_camel2d(float2 x)
{
    float xx = x.x * x.x;
    float yy = x.y * x.y;
    return (4.0f - 2.1f * xx + (xx * xx) / 3.0f) * xx
         + x.x * x.y
         + (-4.0f + 4.0f * yy) * yy;
}

__device__ inline float holder_table2d(float2 x)
{
    float expo = fabsf(1.0f - sqrtf(x.x * x.x + x.y * x.y) / 3.141592653589793f);
    return -fabsf(sinf(x.x) * cosf(x.y) * expf(expo));
}

__device__ inline float evaluate2d(float2 x, unsigned int objective_type)
{
    if (objective_type == 1u) return rastrigin2d(x);
    if (objective_type == 2u) return ackley2d(x);
    if (objective_type == 3u) return himmelblau2d(x);
    if (objective_type == 4u) return six_hump_camel2d(x);
    if (objective_type == 5u) return holder_table2d(x);
    return sphere2d(x);
}

__global__ void pso_update_2d(Particle2D* particles,
                              Params2D params,
                              const unsigned int* assignments,
                              const float2* cluster_best)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= params.count) return;

    Particle2D p = particles[id];

    float f = evaluate2d(p.position, params.objective_type);
    p.fitness = f;
    if (f < p.best_fitness) {
        p.best_fitness = f;
        p.best_position = p.position;
    }

    unsigned int rng = 1103515245u * (id + 1u) + 12345u;
    rng = 1664525u * rng + 1013904223u;
    float r1 = static_cast<float>(rng & 0x00FFFFFFu) / static_cast<float>(0x01000000u);
    rng = 1664525u * rng + 1013904223u;
    float r2 = static_cast<float>(rng & 0x00FFFFFFu) / static_cast<float>(0x01000000u);

    float2 attractor = params.gbest;
    if (params.use_cluster_best != 0u && assignments != nullptr && cluster_best != nullptr) {
        unsigned int cid = assignments[id];
        if (cid < params.cluster_count) {
            attractor = cluster_best[cid];
        }
    }

    float2 inertia = mul2(params.w, p.velocity);
    float2 cognitive = mul2(params.c1 * r1, sub2(p.best_position, p.position));
    float2 social = mul2(params.c2 * r2, sub2(attractor, p.position));
    p.velocity = add2(add2(inertia, cognitive), social);

    p.velocity.x = clampf(p.velocity.x, params.vmin, params.vmax);
    p.velocity.y = clampf(p.velocity.y, params.vmin, params.vmax);

    p.position = add2(p.position, p.velocity);
    p.position.x = clampf(p.position.x, params.xmin, params.xmax);
    p.position.y = clampf(p.position.y, params.xmin, params.xmax);

    particles[id] = p;
}

__global__ void kmeans_assign_2d(const Particle2D* particles,
                                 const float2* centroids,
                                 unsigned int* assignments,
                                 KMeansParams2D params)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= params.particle_count) return;

    float2 p = particles[id].best_position;
    unsigned int best = 0u;
    float2 d0 = sub2(p, centroids[0]);
    float best_dist = dot2(d0, d0);

    for (unsigned int c = 1u; c < params.cluster_count; ++c) {
        float2 d = sub2(p, centroids[c]);
        float dist = dot2(d, d);
        if (dist < best_dist) {
            best_dist = dist;
            best = c;
        }
    }

    assignments[id] = best;
}

__global__ void kmeans_update_centroids_2d(const Particle2D* particles,
                                           const unsigned int* assignments,
                                           float2* centroids,
                                           unsigned int* counts,
                                           KMeansParams2D params)
{
    unsigned int cluster_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (cluster_id >= params.cluster_count) return;

    float2 sum = make_float2(0.0f, 0.0f);
    unsigned int count = 0u;

    for (unsigned int i = 0u; i < params.particle_count; ++i) {
        if (assignments[i] == cluster_id) {
            sum = add2(sum, particles[i].best_position);
            count += 1u;
        }
    }

    counts[cluster_id] = count;
    if (count > 0u) {
        float inv = 1.0f / static_cast<float>(count);
        centroids[cluster_id] = mul2(inv, sum);
    }
}

__device__ inline float rastrigin_nd(const float* x, unsigned int base, unsigned int dim)
{
    const float a = 10.0f;
    const float twopi = 6.283185307179586f;
    float sum = a * static_cast<float>(dim);
    for (unsigned int d = 0u; d < dim; ++d) {
        float v = x[base + d];
        sum += v * v - a * cosf(twopi * v);
    }
    return sum;
}

__device__ inline float ackley_nd(const float* x, unsigned int base, unsigned int dim)
{
    const float a = 20.0f;
    const float b = 0.2f;
    const float c = 6.283185307179586f;
    float sum_sq = 0.0f;
    float sum_cos = 0.0f;
    for (unsigned int d = 0u; d < dim; ++d) {
        float v = x[base + d];
        sum_sq += v * v;
        sum_cos += cosf(c * v);
    }
    float inv_dim = 1.0f / static_cast<float>(dim);
    return -a * expf(-b * sqrtf(sum_sq * inv_dim)) - expf(sum_cos * inv_dim) + a + 2.718281828459045f;
}

__device__ inline float evaluate_nd(const float* x, unsigned int base, unsigned int dim, unsigned int objective_type)
{
    if (objective_type == 102u) return ackley_nd(x, base, dim);
    return rastrigin_nd(x, base, dim);
}

__global__ void pso_update_nd(float* positions,
                              float* velocities,
                              float* best_positions,
                              float* fitness,
                              float* best_fitness,
                              const float* gbest,
                              ParamsND params,
                              const unsigned int* assignments,
                              const float* cluster_best)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= params.count) return;

    unsigned int base = id * params.dim;

    float f = evaluate_nd(positions, base, params.dim, params.objective_type);
    fitness[id] = f;
    if (f < best_fitness[id]) {
        best_fitness[id] = f;
        for (unsigned int d = 0u; d < params.dim; ++d) {
            best_positions[base + d] = positions[base + d];
        }
    }

    unsigned int rng = 1103515245u * (id + 1u) + 12345u;
    unsigned int cid = 0u;
    if (params.use_cluster_best != 0u && assignments != nullptr) {
        cid = assignments[id];
        if (cid >= params.cluster_count) cid = 0u;
    }
    unsigned int cb_base = cid * params.dim;

    for (unsigned int d = 0u; d < params.dim; ++d) {
        rng = 1664525u * rng + 1013904223u;
        float r1 = static_cast<float>(rng & 0x00FFFFFFu) / static_cast<float>(0x01000000u);
        rng = 1664525u * rng + 1013904223u;
        float r2 = static_cast<float>(rng & 0x00FFFFFFu) / static_cast<float>(0x01000000u);

        float attractor = gbest[d];
        if (params.use_cluster_best != 0u && cluster_best != nullptr) {
            attractor = cluster_best[cb_base + d];
        }

        float v = params.w * velocities[base + d]
                + params.c1 * r1 * (best_positions[base + d] - positions[base + d])
                + params.c2 * r2 * (attractor - positions[base + d]);
        v = clampf(v, params.vmin, params.vmax);

        float p = clampf(positions[base + d] + v, params.xmin, params.xmax);
        velocities[base + d] = v;
        positions[base + d] = p;
    }
}

__global__ void kmeans_assign_nd(const float* points,
                                 const float* centroids,
                                 unsigned int* assignments,
                                 KMeansParamsND params)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= params.particle_count) return;

    unsigned int p_base = id * params.dim;
    unsigned int best = 0u;
    float best_dist = CUDART_INF_F;

    for (unsigned int c = 0u; c < params.cluster_count; ++c) {
        unsigned int c_base = c * params.dim;
        float dist = 0.0f;
        for (unsigned int d = 0u; d < params.dim; ++d) {
            float dv = points[p_base + d] - centroids[c_base + d];
            dist += dv * dv;
        }
        if (dist < best_dist) {
            best_dist = dist;
            best = c;
        }
    }

    assignments[id] = best;
}

__global__ void kmeans_update_centroids_nd(const float* points,
                                           const unsigned int* assignments,
                                           float* centroids,
                                           unsigned int* counts,
                                           KMeansParamsND params)
{
    unsigned int cluster_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (cluster_id >= params.cluster_count) return;

    unsigned int c_base = cluster_id * params.dim;
    for (unsigned int d = 0u; d < params.dim; ++d) {
        centroids[c_base + d] = 0.0f;
    }

    unsigned int count = 0u;
    for (unsigned int i = 0u; i < params.particle_count; ++i) {
        if (assignments[i] == cluster_id) {
            unsigned int p_base = i * params.dim;
            for (unsigned int d = 0u; d < params.dim; ++d) {
                centroids[c_base + d] += points[p_base + d];
            }
            count += 1u;
        }
    }

    counts[cluster_id] = count;
    if (count > 0u) {
        float inv = 1.0f / static_cast<float>(count);
        for (unsigned int d = 0u; d < params.dim; ++d) {
            centroids[c_base + d] *= inv;
        }
    }
}
