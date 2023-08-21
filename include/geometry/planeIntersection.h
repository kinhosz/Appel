#ifndef GEOMETRY_PLANEINTERSECTION_H
#define GEOMETRY_PLANEINTERSECTION_H

#include <graphic/color.h>
#include <geometry/vetor.h>

struct PlaneIntersection {
	Color color;
	double distance;
	Vetor normal;

	PlaneIntersection();
	PlaneIntersection(Color color, double distance, Vetor normal);
};

#endif