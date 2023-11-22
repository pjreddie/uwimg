#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "image.h"

float get_pixel(image im, int c, int h, int w)
{   
    if (c < 0){
        c = 0;
    }
    if (w< 0){
        w = 0;
    }
    if (h< 0){
        h = 0;
    }
    if(c >= im.c) {
        c = im.c-1;
    }
    if (w >= im.w) {
        w = im.w-1;
    }
    if(h >= im.h) {
        h = im.h-1;
    }
    return im.data[c*(im.w*im.h) +(h*im.w)+w];
}

void set_pixel(image im, int c, int h, int w, float v)
{
    if (c < 0 || w < 0 || h < 0 || c >= im.c || h >= im.h || w >= im.w){
        return;
    }
    im.data[c*(im.w*im.h) +(h*im.w)+w] = v;
}

image copy_image(image im)
{
    image copy = make_image(im.c, im.h, im.w);
    memcpy(copy.data, im.data, sizeof(float)*im.w*im.h*im.c);
    return copy;
}

image rgb_to_grayscale(image im)
{
    assert(im.c == 3);
    image gray = make_image(1, im.h, im.w);
    // TODO Fill this in
    for (int row = 0; row < im.h; row++) {
        for(int col = 0; col < im.w; col++) {
                float R = 0.299 * get_pixel(im, 0, row, col);
                float G = 0.587 * get_pixel(im, 1, row, col);
                float B = 0.114 * get_pixel(im, 2, row, col); 
                set_pixel(gray, 0, row, col, R+G+B); 
            }
        }
    return gray;
}

void shift_image(image im, int c, float v)
{
    // TODO Fill this in
    for (int row = 0; row < im.h; row++) {
        for(int col = 0; col < im.w; col++) {
                float R = get_pixel(im, c, row, col); 
                set_pixel(im, c, row, col, R+v); 
            }
        }
}

void clamp_image(image im)
{
    // TODO Fill this in
    for (int row = 0; row < im.h; row++) {
        for(int col = 0; col < im.w; col++) {
            for(int channel = 0; channel < im.c; channel++){
                float temp = get_pixel(im, channel, row, col); 
                if(temp > 1){
                    temp = 1;
                } 
                if(temp < 0) {
                    temp = 0;
                }
                set_pixel(im, channel, row, col, temp);
            }
        }
    }
}

// These might be handy
float three_way_max(float a, float b, float c)
{
    return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
}

float three_way_min(float a, float b, float c)
{
    return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}

void rgb_to_hsv(image im)
{
    float H, V, M, S;
    for (int row = 0; row < im.h; row++) {
        for(int col = 0; col < im.w; col++) {
                float R = get_pixel(im, 0, row, col);
                float G = get_pixel(im, 1, row, col);
                float B = get_pixel(im, 2, row, col); 
                V = three_way_max(R,G,B);
                M = three_way_min(R, G, B);
                float C = V - M;
                S = 0;
                if (R != 0 || G != 0 || B != 0) {
                    S = C / V;
                }
                if(C == 0){
                    H = 0;
                } else {
                    if (V == R) {
                        H = (G-B)/C;
                    } else if (V == G) {
                        H = ((B-R)/C) + 2;
                    } else {
                        H = ((R-G)/C) + 4;
                    }
                    if (H < 0) {
                        H = (H/6)+1;
                    } else {
                        H = H/6;
                    }
                }
                set_pixel(im, 0, row, col, H);
                set_pixel(im, 1, row, col, S);
                set_pixel(im, 2, row, col, V);
        }
    }

}

void hsv_to_rgb(image im)
{
    // TODO Fill this in
    float H, M, currR, currG, currB;
    for (int row = 0; row < im.h; row++) {
        for(int col = 0; col < im.w; col++) {
                float R = get_pixel(im, 0, row, col);
                float G = get_pixel(im, 1, row, col);
                float B = get_pixel(im, 2, row, col); 
                float C = G*B;
                M = B-C;
                H=R*6;
                if (H > 5) {
                    H = (R-1)*6;
                }
                if (H < 0) {
                        currR = B;
                        currG = M;
                        currB = M - H * C;
                } else if (H <= 1) {
                        currR = B;
                        currG = M + H * C;
                        currB = M;
                } else if (H <= 3){
                        H = H-2;
                        if(H < 0){
                            currR = M-H*C;
                            currG = B;
                            currB = M;
                        } else {
                            currR = M;
                            currG = B;
                            currB = M + H * C;
                        }
                } else {
                    H = H-4;
                    if (H < 0) {
                            currR = M;
                            currG = M-H*C;
                            currB = B;
                    } else {
                            currR = M+H*C;
                            currG = M;
                            currB = B; 
                    }
                }
                set_pixel(im, 0, row, col, currR);
                set_pixel(im, 1, row, col, currG);
                set_pixel(im, 2, row, col, currB);
        }
    }
}

void scale_image(image im, int c, float v)
{
    // TODO Fill this in
    for (int row = 0; row < im.h; row++) {
        for(int col = 0; col < im.w; col++) {
                float R = get_pixel(im, c, row, col); 
                set_pixel(im, c, row, col, R*v); 
            }
        }
}
