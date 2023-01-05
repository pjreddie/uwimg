#ifdef OPENCV

#include "stdio.h"
#include "stdlib.h"
#include "opencv2/opencv.hpp"
#include "image.h"

using namespace cv;

extern "C" {

    Mat image_to_mat(image im)
    {
        image copy = copy_image(im);
        clamp_image(copy);
        int i,j;
        Mat m(im.h, im.w, CV_8UC3);
        for(j = 0; j < im.h; ++j){
            for(i = 0; i < im.w; ++i){
                m.at<Vec3b>(j, i) = Vec3b( get_pixel(im, i, j, 2)*255,
                        get_pixel(im, i, j, 1)*255,
                        get_pixel(im, i, j, 0)*255);
            }
        }
        return m;
    }

    image mat_to_image(Mat m)
    {
        image im = make_image(m.cols, m.rows, 3);
        int i,j;
        for(j = 0; j < im.h; ++j){
            for(i = 0; i < im.w; ++i){
                Vec3b intensity = m.at<Vec3b>(j, i);
                float blue = intensity.val[0]/255.;
                float green = intensity.val[1]/255.;
                float red = intensity.val[2]/255.;
                set_pixel(im, i, j, 0, red);
                set_pixel(im, i, j, 1, green);
                set_pixel(im, i, j, 2, blue);
            }
        }
        return im;
    }

    void *open_video_stream(const char *f, int c, int w, int h, int fps)
    {
        VideoCapture *cap;
        if(f) cap = new VideoCapture(f);
        else cap = new VideoCapture(c);
        if(!cap->isOpened()) return 0;
        if(w) cap->set(CAP_PROP_FRAME_WIDTH, w);
        if(h) cap->set(CAP_PROP_FRAME_HEIGHT, w);
        if(fps) cap->set(CAP_PROP_FPS, w);
        return (void *) cap;
    }

    image get_image_from_stream(void *p)
    {
        VideoCapture *cap = (VideoCapture *)p;
        Mat m;
        *cap >> m;
        if(m.empty()) return make_image(0,0,0);
        return mat_to_image(m);
    }

    image load_image_cv(char *filename, int channels)
    {
        int flag = -1;
        if (channels == 0) flag = -1;
        else if (channels == 1) flag = 0;
        else if (channels == 3) flag = 1;
        else {
            fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
        }
        Mat m;
        m = imread(filename, flag);
        if(!m.data){
            fprintf(stderr, "Cannot load image \"%s\"\n", filename);
            char buff[256];
            sprintf(buff, "echo %s >> bad.list", filename);
            system(buff);
            return make_image(10,10,3);
            //exit(0);
        }
        image im = mat_to_image(m);
        if (channels) assert(im.c == channels);
        return im;
    }

    int show_image(image im, const char* name, int ms)
    {
        Mat m = image_to_mat(im);
        imshow(name, m);
        int c = waitKey(ms);
        if (c != -1) c = c%256;
        return c;
    }

    void make_window(char *name, int w, int h, int fullscreen)
    {
        namedWindow(name, WINDOW_NORMAL); 
        if (fullscreen) {
            setWindowProperty(name, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
        } else {
            resizeWindow(name, w, h);
            if(strcmp(name, "Demo") == 0) moveWindow(name, 0, 0);
        }
    }

}

#endif
