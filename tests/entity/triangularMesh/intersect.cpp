#include <Appel/entity/triangularMesh.h>
#include <cassert>
using namespace std;
using namespace Appel;

int main(){
    Color red(1, 0, 0), green(0, 1, 0), blue(0, 0, 1), white(1, 1, 1);
    
    Point p0(1, 0, 0), p1(0, 1, 0), p2(-1, 0, 0), p3(0, -1, 0), p4(0, 0, 1);
    
    Vetor n1 = (Vetor(p1) - Vetor(p0)).cross(Vetor(p4) - Vetor(p0)).normalize();
    Vetor n2 = (Vetor(p2) - Vetor(p1)).cross(Vetor(p4) - Vetor(p1)).normalize();
    Vetor n3 = (Vetor(p3) - Vetor(p2)).cross(Vetor(p4) - Vetor(p2)).normalize();
    Vetor n4 = (Vetor(p0) - Vetor(p3)).cross(Vetor(p4) - Vetor(p3)).normalize();

    Vetor aux(0, 0, 1);

    Triangle t1(p0, p1, p4, aux, aux, aux, n1, red);
    Triangle t2(p1, p2, p4, aux, aux, aux, n2, green);
    Triangle t3(p2, p3, p4, aux, aux, aux, n3, blue);
    Triangle t4(p3, p0, p4, aux, aux, aux, n4, white);

    vector<Triangle> triangles;
    triangles.push_back(t1);
    triangles.push_back(t2);
    triangles.push_back(t3);
    triangles.push_back(t4);

    TriangularMesh tMesh(triangles);

    tMesh.setPhongValues(0, 0, 0, 0, 0, 1.0);

    Point observer1(2, 2, 0.5);
    Point observer2(-2, 2, 0.5);
    Point observer3(-2, -2, 0.5);
    Point observer4(2, -2, 0.5);

    Vetor dir1(-1, -1, 0);
    Vetor dir2(1, -1, 0);
    Vetor dir3(1, 1, 0);
    Vetor dir4(-1, 1, 0);

    Ray r1(observer1, dir1.normalize());
    Ray r2(observer2, dir2.normalize());
    Ray r3(observer3, dir3.normalize());
    Ray r4(observer4, dir4.normalize());

    SurfaceIntersection sf1, sf2, sf3, sf4;

    sf1 = tMesh.intersect(r1);
    sf2 = tMesh.intersect(r2);
    sf3 = tMesh.intersect(r3);
    sf4 = tMesh.intersect(r4);

    assert(sf1.color == red);
    assert(sf2.color == green);
    assert(sf3.color == blue);
    assert(sf4.color == white);
}
