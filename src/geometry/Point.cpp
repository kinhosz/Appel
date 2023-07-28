#include <Point.h>
#include <cmath>

Point::Point() : x(0), y(0) {}

Point::Point(double x, double y) : x(x), y(y) {}

double Point::distance(const Point &other) {
    return sqrt(pow(x - other.x, 2) + pow(y - other.y, 2));
}

// 1. #include <Point.h>
//    This includes the header file for the Point class. The header file
//    contains the declarations of the functions and the struct definition.
// 2. #include <cmath>
//    This includes the cmath header, which contains the sqrt and pow
//    functions.
// 3. Point::Point() : x(0), y(0) {}
//    This is the definition of the default constructor. The Point:: part
//    means that the function is a member of the Point struct. The : x(0),
//    y(0) part is the initializer list, which initializes the members with
//    the given values.
// 4. Point::Point(double x, double y) : x(x), y(y) {}
//    This is the definition of the other constructor. It takes two arguments,
//    x and y, and initializes the members with them.
// 5. double Point::distance(const Point &other) {
//    return sqrt(pow(x - other.x, 2) + pow(y - other.y, 2));
//    }
//    This is the definition of the distance function. It takes a const
//    reference to another Point as an argument, and returns a double. The
//    const keyword means that the function does not modify the argument.
//    The & means that the argument is passed by reference, which means that
//    the function can access the original object instead of a copy. This
//    is more efficient than passing by value, and allows the function to
//    modify the object if it is not const.
// 6. return sqrt(pow(x - other.x, 2) + pow(y - other.y, 2));
//    This returns the distance between the two points. The sqrt and pow
//    functions are defined in the cmath header.