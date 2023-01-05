#ifndef IMAGE_H
#define IMAGE_H
#include <stdio.h>

#include "matrix.h"
#define TWOPI 6.2831853

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#ifdef __cplusplus
extern "C" {
#endif

// DO NOT CHANGE THIS FILE

typedef struct{
    int c,h,w;
    float *data;
} image;

// A 2d point.
// float x, y: the coordinates of the point.
typedef struct{
    float x, y;
} point;

// A descriptor for a point in an image.
// point p: x,y coordinates of the image pixel.
// int n: the number of floating point values in the descriptor.
// float *data: the descriptor for the pixel.
typedef struct{
    point p;
    int n;
    float *data;
} descriptor;

// A match between two points in an image.
// point p, q: x,y coordinates of the two matching pixels.
// int ai, bi: indexes in the descriptor array. For eliminating duplicates.
// float distance: the distance between the descriptors for the points.
typedef struct{
    point p, q;
    int ai, bi;
    float distance;
} match;

// Basic operations
float get_pixel(image im, int c, int h, int w);
void set_pixel(image im, int c, int h, int w, float v);
image copy_image(image im);
image rgb_to_grayscale(image im);
image grayscale_to_rgb(image im, float r, float g, float b);
void rgb_to_hsv(image im);
void hsv_to_rgb(image im);
void shift_image(image im, int c, float v);
void scale_image(image im, int c, float v);
void clamp_image(image im);
image get_channel(image im, int c);
int same_image(image a, image b, float eps);
image sub_image(image a, image b);
image add_image(image a, image b);

// Loading and saving
image make_image(int c, int h, int w);
image load_image(char *filename);
void save_image(image im, const char *name);
void save_png(image im, const char *name);
void save_image_binary(image im, const char *fname);
image load_image_binary(const char *fname);
void save_png(image im, const char *name);
void free_image(image im);

// Resizing
float nn_interpolate(image im, int c, float h, float w);
image nn_resize(image im, int h, int w);
float bilinear_interpolate(image im, int c, float h, float w);
image bilinear_resize(image im, int h, int w);

// Filtering
image convolve_image(image im, image filter, int preserve);
image make_box_filter(int w);
image make_highpass_filter();
image make_sharpen_filter();
image make_emboss_filter();
image make_gaussian_filter(float sigma);
image make_gx_filter();
image make_gy_filter();
void feature_normalize(image im);
void l1_normalize(image im);
void threshold_image(image im, float thresh);
image *sobel_image(image im);
image colorize_sobel(image im);
image smooth_image(image im, float sigma);

// Harris and Stitching
point make_point(float x, float y);
point project_point(matrix H, point p);
matrix compute_homography(match *matches, int n);
image structure_matrix(image im, float sigma);
image cornerness_response(image S);
void free_descriptors(descriptor *d, int n);
image cylindrical_project(image im, float f);
void mark_corners(image im, descriptor *d, int n);
image find_and_draw_matches(image a, image b, float sigma, float thresh, int nms);
void detect_and_draw_corners(image im, float sigma, float thresh, int nms);
int model_inliers(matrix H, match *m, int n, float thresh);
image combine_images(image a, image b, matrix H);
match *match_descriptors(descriptor *a, int an, descriptor *b, int bn, int *mn);
descriptor *harris_corner_detector(image im, float sigma, float thresh, int nms, int *n);
image panorama_image(image a, image b, float sigma, float thresh, int nms, float inlier_thresh, int iters, int cutoff);

// Optical Flow
image make_integral_image(image im);
image box_filter_image(image im, int s);
image time_structure_matrix(image im, image prev, int s);
image velocity_image(image S, int stride);
image optical_flow_images(image im, image prev, int smooth, int stride);
void optical_flow_webcam(int smooth, int stride, int div);
void draw_flow(image im, image v, float scale);

#ifdef OPENCV
void *open_video_stream(const char *f, int c, int h, int w, int fps);
image get_image_from_stream(void *p);
void make_window(char *name, int h, int w, int fullscreen);
int show_image(image im, const char *name, int ms);
#endif

// Machine Learning

typedef enum{LINEAR, LOGISTIC, RELU, LRELU, SOFTMAX} ACTIVATION;

typedef struct {
    matrix in;              // Saved input to a layer
    matrix w;               // Current weights for a layer
    matrix dw;              // Current weight updates
    matrix v;               // Past weight updates (for use with momentum)
    matrix out;             // Saved output from the layer
    ACTIVATION activation;  // Activation the layer uses
} layer;

typedef struct{
    matrix X;
    matrix y;
} data;

typedef struct {
    layer *layers;
    int n;
} model;

data load_classification_data(char *images, char *label_file, int bias);
void free_data(data d);
data random_batch(data d, int n);
char *fgetl(FILE *fp);
void activate_matrix(matrix m, ACTIVATION a);
void gradient_matrix(matrix m, ACTIVATION a, matrix d);
matrix forward_layer(layer *l, matrix in);
matrix backward_layer(layer *l, matrix delta);
void update_layer(layer *l, double rate, double momentum, double decay);
layer make_layer(int input, int output, ACTIVATION activation);
matrix load_matrix(const char *fname);
void save_matrix(matrix m, const char *fname);

#ifdef __cplusplus
}
#endif
#endif

