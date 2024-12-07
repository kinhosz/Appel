#include <geometry/size.h>
#include <cassert>

int main() {
    Size s1, s2(1920, 1080);

    assert(s1.width == 0 && s1.height == 0);
    assert(s2.width == 1920 && s2.height == 1080);

    s1.setWidth(800);
    s1.setHeight(600);

    assert(s1 == Size(800, 600));
    assert(s2 == Size(1920, 1080));
    assert(s1 != s2);
}

