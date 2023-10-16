#include <gpu/helper.h>

__device__ int isInside(GPoint point, GTriangle triangle) {
    float matrix[4][4];
    
    for(int j=0;j<3;j++) {
        matrix[0][j] = triangle.point[j].x;
        matrix[1][j] = triangle.point[j].y;
        matrix[2][j] = triangle.point[j].z;
        matrix[3][j] = 1.0;
    }

    matrix[0][3] = point.x;
    matrix[1][3] = point.y;
    matrix[2][3] = point.z;
    matrix[3][3] = 1.0;

    /* gauss */
    int n = 4, m = 4;

    for(int k=0;k<m-1;k++){
        int pivot = -1;
        for(int i=k;i<n;i++){
            if(f_cmp(matrix[i][k], 0) == 0) continue;

            pivot = i;
            break;
        }
        if(pivot == -1) break;

        double f = matrix[pivot][k];

        for(int j=0;j<m;j++) matrix[pivot][j] /= f;

        for(int i=0;i<n;i++){
            f = matrix[i][k];

            if(i == pivot) continue;

            for(int j=0;j<m;j++) matrix[i][j] -= f * matrix[pivot][j];
        }

        for(int j=0;j<m;j++) {
            float tmp = matrix[k][j];
            matrix[k][j] = matrix[pivot][j];
            matrix[pivot][j] = tmp;
        }
    }

    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            if(i >= m-1 && f_cmp(matrix[i][j], 0) != 0) return 0;
            if(i < m-1 && j == i && f_cmp(matrix[i][j], 1) != 0) return 0;
            if(i < m-1 && j < m-1 && j != i && f_cmp(matrix[i][j], 0) != 0) return 0;
        }
    }

    float sumt = 0.0;
    for(int i=0;i<m-1;i++) {
        sumt += matrix[i][m-1];
        if(f_cmp(matrix[i][m-1], 0.0) == -1 ||
            f_cmp(matrix[i][m-1], 1.0) == 1) return 0;
    }

    if(f_cmp(sumt, 1.0) != 0) return 0;

    return 1;
}
