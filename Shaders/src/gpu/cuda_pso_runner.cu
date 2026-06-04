#include "PSO.cu"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void checkCuda(cudaError_t err, const char* what)
{
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

std::vector<Particle2D> initializeParticles(int count, float xmin, float xmax, float vmin, float vmax, unsigned int seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> pos(xmin, xmax);
    std::uniform_real_distribution<float> vel(vmin, vmax);

    std::vector<Particle2D> particles(count);
    for (auto& p : particles) {
        p.position = make_float2(pos(rng), pos(rng));
        p.velocity = make_float2(vel(rng), vel(rng));
        p.best_position = p.position;
        p.fitness = std::numeric_limits<float>::infinity();
        p.best_fitness = std::numeric_limits<float>::infinity();
    }

    return particles;
}

float2 bestPosition(const std::vector<Particle2D>& particles)
{
    auto best = std::min_element(
        particles.begin(),
        particles.end(),
        [](const Particle2D& a, const Particle2D& b) {
            return a.best_fitness < b.best_fitness;
        });

    if (best == particles.end()) {
        return make_float2(0.0f, 0.0f);
    }
    return best->best_position;
}

float bestFitness(const std::vector<Particle2D>& particles)
{
    auto best = std::min_element(
        particles.begin(),
        particles.end(),
        [](const Particle2D& a, const Particle2D& b) {
            return a.best_fitness < b.best_fitness;
        });

    return best == particles.end() ? std::numeric_limits<float>::infinity() : best->best_fitness;
}

int readIntArg(int argc, char** argv, const std::string& flag, int defaultValue)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == flag) {
            return std::atoi(argv[i + 1]);
        }
    }
    return defaultValue;
}

unsigned int readUIntArg(int argc, char** argv, const std::string& flag, unsigned int defaultValue)
{
    return static_cast<unsigned int>(readIntArg(argc, argv, flag, static_cast<int>(defaultValue)));
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const int count = readIntArg(argc, argv, "--particles", 2048);
        const int iterations = readIntArg(argc, argv, "--iterations", 500);
        const unsigned int seed = readUIntArg(argc, argv, "--seed", 42u);
        const unsigned int objective = readUIntArg(argc, argv, "--objective", 0u);

        const float xmin = -5.0f;
        const float xmax = 5.0f;
        const float vmin = -1.0f;
        const float vmax = 1.0f;

        std::vector<Particle2D> particles = initializeParticles(count, xmin, xmax, vmin, vmax, seed);

        Particle2D* deviceParticles = nullptr;
        checkCuda(cudaMalloc(&deviceParticles, sizeof(Particle2D) * particles.size()), "cudaMalloc particles");
        checkCuda(cudaMemcpy(deviceParticles, particles.data(), sizeof(Particle2D) * particles.size(), cudaMemcpyHostToDevice),
                  "cudaMemcpy particles H2D");

        const int threads = 256;
        const int blocks = (count + threads - 1) / threads;

        Params2D params{};
        params.w = 0.729f;
        params.c1 = 1.49445f;
        params.c2 = 1.49445f;
        params.count = static_cast<unsigned int>(count);
        params.xmin = xmin;
        params.xmax = xmax;
        params.vmin = vmin;
        params.vmax = vmax;
        params.objective_type = objective;
        params.cluster_count = 0u;
        params.use_cluster_best = 0u;

        for (int iter = 0; iter < iterations; ++iter) {
            params.gbest = bestPosition(particles);

            pso_update_2d<<<blocks, threads>>>(deviceParticles, params, nullptr, nullptr);
            checkCuda(cudaGetLastError(), "pso_update_2d launch");
            checkCuda(cudaDeviceSynchronize(), "pso_update_2d synchronize");
            checkCuda(cudaMemcpy(particles.data(), deviceParticles, sizeof(Particle2D) * particles.size(), cudaMemcpyDeviceToHost),
                      "cudaMemcpy particles D2H");
        }

        float2 best = bestPosition(particles);
        std::cout << "CUDA PSO complete\n";
        std::cout << "Particles: " << count << ", iterations: " << iterations << "\n";
        std::cout << "Objective type: " << objective << " (0=sphere, 1=rastrigin, 2=ackley, 3=himmelblau, 4=six-hump, 5=holder)\n";
        std::cout << "Best fitness: " << bestFitness(particles) << "\n";
        std::cout << "Best position: " << best.x << " " << best.y << "\n";

        checkCuda(cudaFree(deviceParticles), "cudaFree particles");
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "CUDA PSO runner failed: " << ex.what() << "\n";
        return 1;
    }
}
