#pragma once

void Render(int screenWidth, int screenHeight, const unsigned int numFrames, const unsigned int samplesPerPixel, const unsigned int threadsPerBlock, const unsigned int maxDepth, float* backbuffer, unsigned long long& outRayCount);
