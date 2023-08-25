#include <geometry/triangle.h>
#include <geometry/point.h>
#include <graphic/color.h>
#include <geometry/vetor.h> 
#include <geometry/utils.h>
#include <cmath>

const double EPSILON = 1e-12;

int cmp(double a, double b){
    if(std::abs(a - b) < EPSILON) return 0;
    return (a < b ? -1 : 1);
}

std::vector<Triangle> createTriangles(
    std::vector<Point>::size_type numberOfTriangles,
    std::vector<Point> vertices,
    std::vector<std::array<int, 3>> triangles,
    std::vector<Vetor> vertexNormals,  
    std::vector<Vetor> triangleNormals, 
    std::vector<Color> colors
) {
    std::vector<Triangle> triangleObjects;
    for (std::vector<Point>::size_type i = 0; i < numberOfTriangles; ++i) {
        const std::array<int, 3>& indices = triangles[i];
        Triangle triangle(
            vertices[indices[0]],
            vertices[indices[1]],
            vertices[indices[2]],
            vertexNormals[indices[0]],  
            vertexNormals[indices[1]],
            vertexNormals[indices[2]],
            triangleNormals[i], 
            colors[i]
        );
        triangleObjects.push_back(triangle);
    }
    return triangleObjects;
}

std::vector<double> gaussElimination(std::vector<std::vector<double>> matrix) {
    int n = matrix.size();
    int m = (n == 0? 0 : matrix[0].size());

    for(int k=0;k<m-1;k++){
        int pivot = -1;
        for(int i=k;i<n;i++){
            if(cmp(matrix[i][k], 0) == 0) continue;

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

        swap(matrix[k], matrix[pivot]);
    }

    std::vector<double> ret;

    bool hasUniqueSolution = true;

    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            if(i >= m-1 && cmp(matrix[i][j], 0) != 0) hasUniqueSolution = false;
            if(i < m-1 && j == i && cmp(matrix[i][j], 1) != 0) hasUniqueSolution = false;
            if(i < m-1 && j < m-1 && j != i && cmp(matrix[i][j], 0) != 0) hasUniqueSolution = false;
        }
    }

    if(!hasUniqueSolution) return ret;

    for(int i=0;i<m-1;i++) ret.push_back(matrix[i][m-1]);

    return ret;
}
