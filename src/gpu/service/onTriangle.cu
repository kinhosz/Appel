#include <gpu/service.h>

__device__ bool hasValidBaricenter(float matrix[][4]) {
    int n = 4, m = 4;

    for(int k=0;k<m-1;k++) {
        int pivot = -1;
        for(int i=k;i<n;i++) {
            if(abs(matrix[i][k]) < 1e-9) continue;
            pivot = i;
            break;
        }
        if(pivot == -1) break;

        double f = matrix[pivot][k];

        for(int j=0;j<m;j++) matrix[pivot][j] /= f;

        for(int i=0;i<n;i++) {
            f = matrix[i][k];
            if(i == pivot) continue;
            for(int j=0;j<m;j++) matrix[i][j] -= f * matrix[pivot][j];
        }

        for(int j=0;j<4;j++){
            float tmp = matrix[k][j];
            matrix[k][j] = matrix[pivot][j];
            matrix[pivot][j] = tmp;
        }
    }

    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            if(i >= m-1 && abs(matrix[i][j]) > 1e-9) return false;
            if(i < m-1 && j == i && abs(matrix[i][j] - 1.0) > 1e-9) return false;
            if(i < m-1 && j < m-1 && j != i && abs(matrix[i][j]) > 1e-9) return false;
        }
    }

    float sum_bar = 0;

    for(int i=0;i<m-1;i++) {
        if(matrix[i][m-1] < 0.0 || matrix[i][m-1] > 1.0) return false;
        sum_bar += matrix[i][m-1];
    }

    if(abs(sum_bar - 1.0) > 1e-9) return false;

    return true;
}

__device__ bool onTriangle(GTriangle gt, GRay ray, float t) {
    float px = ray.lx + ray.dx * t;
    float py = ray.ly + ray.dy * t;
    float pz = ray.lz + ray.dz * t;

    float matrix[4][4];
    matrix[0][0] = gt.p0x, matrix[0][1] = gt.p1x, matrix[0][2] = gt.p2x, matrix[0][3] = px;
    matrix[1][0] = gt.p0y, matrix[1][1] = gt.p1y, matrix[1][2] = gt.p2y, matrix[1][3] = py;
    matrix[2][0] = gt.p0z, matrix[2][1] = gt.p1z, matrix[2][2] = gt.p2z, matrix[2][3] = pz;
    matrix[3][0] = 1.0, matrix[3][1] = 1.0, matrix[3][2] = 1.0, matrix[3][3] = 1.0;

    return hasValidBaricenter(matrix);
}
