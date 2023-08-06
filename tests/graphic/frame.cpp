#include <graphic/frame.h>
#include <cassert>

int main(){
    int vRes = 100;
    int hRes = 200;

    Frame frame(vRes, hRes);

    assert(vRes == frame.vertical());
    assert(hRes == frame.horizontal());

    for(int i=0;i<vRes;i++){
        for(int j=0;j<hRes;j++){
            int id = i * vRes + j;

            int red = id%255;
            id /= 255;
            int green = id%255;
            id /= 255;
            int blue = id%255;

            frame.setPixel(i, j, Pixel(red, green, blue));
        }
    }

    for(int i=0;i<vRes;i++){
        for(int j=0;j<hRes;j++){
            int id = i * vRes + j;

            int red = id%255;
            id /= 255;
            int green = id%255;
            id /= 255;
            int blue = id%255;

            Pixel pixel = frame.getPixel(i, j);

            assert(pixel.red == red);
            assert(pixel.green == green);
            assert(pixel.blue == blue);
        }
    }
}
