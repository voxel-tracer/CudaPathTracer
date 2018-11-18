#include <stdint.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <vector>

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "../Source/Config.h"
#include "../Source/Test.h"

static float* g_Backbuffer;

void write_image(const char* output_file) {
    char *data = new char[kBackbufferWidth * kBackbufferHeight * 3];
    int idx = 0;
    for (int y = kBackbufferHeight - 1; y >= 0; y--) {
        for (int x = 0; x < kBackbufferWidth; x++) {
            const float * backbuffer = g_Backbuffer + (y*kBackbufferWidth + x) * 4;
            data[idx++] = std::min(255, int(255.99*backbuffer[0]));
            data[idx++] = std::min(255, int(255.99*backbuffer[1]));
            data[idx++] = std::min(255, int(255.99*backbuffer[2]));
        }
    }
    stbi_write_png(output_file, kBackbufferWidth, kBackbufferHeight, 3, (void*)data, kBackbufferWidth * 3);
    delete[] data;
}

float render(const int numFrames, const int samplesPerPixel, const int threadsPerBlock)
{
    unsigned long long rayCounter = 0;

    const clock_t start_time = clock();

    Render(kBackbufferWidth, kBackbufferHeight, numFrames, samplesPerPixel, threadsPerBlock, g_Backbuffer, rayCounter);

    const float duration = (float)(clock() - start_time) / CLOCKS_PER_SEC;
    const float throughput = rayCounter / duration * 1.0e-6f;
    //printf("   total %llu rays in %.2fs (%.1fMrays/s)\n", rayCounter, duration, throughput);
    return throughput;
}

int main(int argc, char** argv) {
    const int frames[] = { 200 };
    const int numFrames = sizeof(frames) / sizeof(int);
    const int samples[] = { 16, 32 };
    const int numSamples = sizeof(samples) / sizeof(int);
    const int threads[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    const int numThreads = sizeof(threads) / sizeof(int);

    const bool compute_median = true;

    g_Backbuffer = new float[kBackbufferWidth * kBackbufferHeight * 4];
    memset(g_Backbuffer, 0, kBackbufferWidth * kBackbufferHeight * 4 * sizeof(g_Backbuffer[0]));

    for (int f = 0; f < numFrames; f++)
    {
        for (int s = 0; s < numSamples; s++)
        {
            for (int t = 0; t < numThreads; t++)
            {
                int num_threads = threads[t] * 32;
                printf("%d frames, %d samples, %d threads\n", frames[f], samples[s], num_threads);

                if (compute_median) {
                    std::vector<float> v;

                    for (int i = 0; i < 10; i++)
                    {
                        float throughput = render(frames[f], samples[s], num_threads);
                        fflush(stdout);

                        v.push_back(throughput);
                    }

                    std::sort(v.begin(), v.end());
                    float median = (v[5] + v[6]) / 2;
                    printf("  median throughput %.1fM rays/s\n", median);
                }
                else {
                    render(frames[f], samples[s], num_threads);
                }
            }
        }
    }

    //write_image("image.png");

    return 0;
}
