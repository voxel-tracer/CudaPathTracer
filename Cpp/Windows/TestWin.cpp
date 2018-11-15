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

float render(const unsigned int numFrames, const unsigned int samplesPerPixel, const unsigned int threadsPerBlock)
{
    unsigned long rayCounter = 0;

    const clock_t start_time = clock();

    Render(kBackbufferWidth, kBackbufferHeight, numFrames, samplesPerPixel, threadsPerBlock, g_Backbuffer, rayCounter);

    const float duration = (float)(clock() - start_time) / CLOCKS_PER_SEC;
    const float throughput = rayCounter / duration * 1.0e-6f;
    //printf("   total %lu rays in %.2fs (%.1fMrays/s)\n", rayCounter, duration, throughput);
    return throughput;
}

int main(int argc, char** argv) {
    unsigned int numFrames[3] = { 100, 200, 400 };
    unsigned int numSamples[6] = { 1, 2, 4, 8, 16, 32 };
    unsigned int numThreads[8] = { 32, 64, 96, 128, 160, 182, 224, 256 };

    g_Backbuffer = new float[kBackbufferWidth * kBackbufferHeight * 4];
    memset(g_Backbuffer, 0, kBackbufferWidth * kBackbufferHeight * 4 * sizeof(g_Backbuffer[0]));

    for (int frame = 0; frame < 3; frame++)
    {
        for (int sample = 0; sample < 6; sample++)
        {
            for (int threads = 0; threads < 8; threads++)
            {
                printf("%d frames, %d samples, %d threads\n", numFrames[frame], numSamples[sample], numThreads[threads]);

                std::vector<float> v;

                for (int i = 0; i < 10; i++)
                {
                    float throughput = render(numFrames[frame], numSamples[sample], numThreads[threads]);
                    fflush(stdout);

                    v.push_back(throughput);
                }

                std::sort(v.begin(), v.end());
                float median = (v[5] + v[6]) / 2;
                printf("  median throughput %.1fM rays/s\n", median);
            }
        }
    }

    //write_image("image.png");

    return 0;
}
