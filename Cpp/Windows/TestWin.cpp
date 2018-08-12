#include <stdint.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "../Source/Config.h"
#include "../Source/Test.h"

static size_t RenderFrame();

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

int main(int argc, char** argv) {
    g_Backbuffer = new float[kBackbufferWidth * kBackbufferHeight * 4];
    memset(g_Backbuffer, 0, kBackbufferWidth * kBackbufferHeight * 4 * sizeof(g_Backbuffer[0]));

    // Main rendering loop
    const clock_t start_time = clock();
    int rayCounter = 0;

    Render(kBackbufferWidth, kBackbufferHeight, g_Backbuffer, rayCounter);

    const float duration = (float) (clock() - start_time) / CLOCKS_PER_SEC;
    printf("total %d rays in %.2fs (%.1fMrays/s)\n", rayCounter, duration, rayCounter / duration * 1.0e-6f);

    write_image("image.png");

    return 0;
}
