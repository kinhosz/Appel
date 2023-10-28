#include <gpu/kernel.h>
#include <gpu/helper.h>
#include <stdio.h>

__global__ void castRay(GRay *rays, int *rays_N, GTriangleArray *cache, int *triangles_N, float *res_dist, int *table) {
    //clock_t t = clock();
    int ray_id = blockIdx.x;
    int ty = threadIdx.y + (blockDim.y * blockIdx.y);

    GRay ray = rays[ray_id];
    GTriangle triangle;
    triangle.host_id = cache->host_id[ty];
    for(int i=0;i<3;i++) {
        triangle.point[i].x = cache->point[i].x[ty];
        triangle.point[i].y = cache->point[i].y[ty];
        triangle.point[i].z = cache->point[i].z[ty];
    }

    res_dist[ty] = triangleIntersect(ray, triangle);

    //t = clock() - t;
    //if(threadIdx.y%32 == 0) table[blockIdx.y * (blockDim.y / 32) + threadIdx.y/32] = t;
}

/*
min[13300]. max[209077]. avg[26662.792969]
*/

/*

GTriangle {
    GPoint point[3] {
        float x, y, z;
    }
    int host_id;
}

GTriangle -> point[0], point[1], point[2], host_id

[x0, y0, z0, x1, y1, z1, x2, y2, z2, host_id][x0, y0, z0, x1, y1, z1, x2, y2, z2, host_id]...


G[i].point[j].y
G[i].host_id

on cpu...

G.host_id[i]
G.point[j].y[i]

*/