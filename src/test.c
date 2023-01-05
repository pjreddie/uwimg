#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include "matrix.h"
#include "image.h"
#include "test.h"
#include "args.h"


float avg_diff(image a, image b)
{
    float diff = 0;
    int i;
    for(i = 0; i < a.w*a.h*a.c; ++i){
        diff += b.data[i] - a.data[i];
    }
    return diff/(a.w*a.h*a.c);
}

image center_crop(image im)
{
    image c = make_image(im.c, im.h/2, im.w/2);
    int i, j, k;
    for(k = 0; k < im.c; ++k){
        for(j = 0; j < im.h/2; ++j){
            for(i = 0; i < im.w/2; ++i){
                set_pixel(c, k, j, i, get_pixel(im, k, j+im.h/4, i+im.w/4));
            }
        }
    }
    return c;
}

void feature_normalize2(image im)
{
    int i;
    float min = im.data[0];
    float max = im.data[0];
    for(i = 0; i < im.w*im.h*im.c; ++i){
        if(im.data[i] > max) max = im.data[i];
        if(im.data[i] < min) min = im.data[i];
    }
    for(i = 0; i < im.w*im.h*im.c; ++i){
        im.data[i] = (im.data[i] - min)/(max-min);
    }
}

int tests_total = 0;
int tests_fail = 0;

int within_eps(float a, float b, float eps){
    return a-eps<b && b<a+eps;
}

int same_point(point p, point q, float eps)
{
    return within_eps(p.x, q.x, eps) && within_eps(p.y, q.y, eps);
}

int same_matrix(matrix m, matrix n)
{
    if(m.rows != n.rows || m.cols != n.cols) return 0;
    int i,j;
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            if(!within_eps(m.data[i][j], n.data[i][j], EPS)) return 0;
        }
    }
    return 1;
}

int same_image(image a, image b, float eps)
{
    int i;
    if(a.w != b.w || a.h != b.h || a.c != b.c) {
        printf("    Expected %d x %d x %d image, got %d x %d x %d\n", b.w, b.h, b.c, a.w, a.h, a.c);
        return 0;
    }
    for(i = 0; i < a.w*a.h*a.c; ++i){
        int x = i % a.w;
        int y = (i / a.w) % a.h;
        int z = (i / (a.w * a.h));
        float thresh = (fabs(b.data[i]) + fabs(a.data[i])) * eps / 2;
        if (thresh > eps) eps = thresh;
        if(!within_eps(a.data[i], b.data[i], eps)) 
        {
            printf("    Index %d, Pixel (%d, %d, %d) should be %f, but it is %f! \n", i, x, y, z, b.data[i], a.data[i]);
            return 0;
        }
    }
    return 1;
}

void make_hw0_test()
{
    image dots = make_image(3, 2, 4);
    set_pixel(dots, 0, 0, 0, 0/255.);
    set_pixel(dots, 1, 0, 0, 1/255.);
    set_pixel(dots, 2, 0, 0, 2/255.);
                            
    set_pixel(dots, 0, 0, 1, 255/255.);
    set_pixel(dots, 1, 0, 1, 3/255.);
    set_pixel(dots, 2, 0, 1, 4/255.);
                            
    set_pixel(dots, 0, 0, 2, 5/255.);
    set_pixel(dots, 1, 0, 2, 254/255.);
    set_pixel(dots, 2, 0, 2, 6/255.);
                            
    set_pixel(dots, 0, 0, 3, 7/255.);
    set_pixel(dots, 1, 0, 3, 8/255.);
    set_pixel(dots, 2, 0, 3, 253/255.);
                            
    set_pixel(dots, 0, 1, 0, 252/255.);
    set_pixel(dots, 1, 1, 0, 251/255.);
    set_pixel(dots, 2, 1, 0, 250/255.);
                            
    set_pixel(dots, 0, 1, 1, 9/255.);
    set_pixel(dots, 1, 1, 1, 249/255.);
    set_pixel(dots, 2, 1, 1, 248/255.);
                            
    set_pixel(dots, 0, 1, 2, 247/255.);
    set_pixel(dots, 1, 1, 2, 10/255.);
    set_pixel(dots, 2, 1, 2, 246/255.);
                            
    set_pixel(dots, 0, 1, 3, 245/255.);
    set_pixel(dots, 1, 1, 3, 244/255.);
    set_pixel(dots, 2, 1, 3, 11/255.);
    save_png(dots, "data/dotz");
}

void test_get_pixel(){
    image im = load_image("data/dotz.png");
    // Within image
    {
        // Top left
        TEST(within_eps(0,      get_pixel(im, 0,0,0), EPS));
        TEST(within_eps(1./255, get_pixel(im, 1,0,0), EPS));
        TEST(within_eps(2./255, get_pixel(im, 2,0,0), EPS));

        TEST(within_eps(255./255, get_pixel(im, 0,0,1), EPS));
        TEST(within_eps(3./255, get_pixel(im, 1,0,1), EPS));
        TEST(within_eps(4./255, get_pixel(im, 2,0,1), EPS));

        // Bottom right
        TEST(within_eps(247./255, get_pixel(im, 0,1,2), EPS));
        TEST(within_eps(10./255,  get_pixel(im, 1,1,2), EPS));
        TEST(within_eps(246./255, get_pixel(im, 2,1,2), EPS));
                                                     
        TEST(within_eps(245./255, get_pixel(im, 0,1,3), EPS));
        TEST(within_eps(244./255, get_pixel(im, 1,1,3), EPS));
        TEST(within_eps(11./255,  get_pixel(im, 2,1,3), EPS));
    }
    // Outside image
    {
        TEST(within_eps(1./255, get_pixel(im, 1,-1,-1), EPS));
        TEST(within_eps(4./255, get_pixel(im, 2,-1, 1), EPS));
        TEST(within_eps(7./255, get_pixel(im, 0, 0, 4), EPS));

        TEST(within_eps(11./255, get_pixel(im, 2,2,4), EPS));
        TEST(within_eps(246./255,get_pixel(im, 2,2,2), EPS));
    }

    free_image(im);
}

void test_set_pixel(){
    image gt = load_image("data/dotz.png");
    image d = make_image(3,2,4);
    set_pixel(d, 0, 0, 0, 0/255.);
    set_pixel(d, 1, 0, 0, 1/255.);
    set_pixel(d, 2, 0, 0, 2/255.);
                         
    set_pixel(d, 0, 0, 1, 255/255.);
    set_pixel(d, 1, 0, 1, 3/255.);
    set_pixel(d, 2, 0, 1, 4/255.);
                         
    set_pixel(d, 0, 0, 2, 5/255.);
    set_pixel(d, 1, 0, 2, 254/255.);
    set_pixel(d, 2, 0, 2, 6/255.);
                         
    set_pixel(d, 0, 0, 3, 7/255.);
    set_pixel(d, 1, 0, 3, 8/255.);
    set_pixel(d, 2, 0, 3, 253/255.);
                         
    set_pixel(d, 0, 1, 0, 252/255.);
    set_pixel(d, 1, 1, 0, 251/255.);
    set_pixel(d, 2, 1, 0, 250/255.);
                         
    set_pixel(d, 0, 1, 1, 9/255.);
    set_pixel(d, 1, 1, 1, 249/255.);
    set_pixel(d, 2, 1, 1, 248/255.);
                         
    set_pixel(d, 0, 1, 2, 247/255.);
    set_pixel(d, 1, 1, 2, 10/255.);
    set_pixel(d, 2, 1, 2, 246/255.);
                         
    set_pixel(d, 0, 1, 3, 245/255.);
    set_pixel(d, 1, 1, 3, 244/255.);
    set_pixel(d, 2, 1, 3, 11/255.);
    // Test images are same
    TEST(same_image(d, gt, EPS));
    free_image(gt);
    free_image(d);
}

void test_grayscale()
{
    image im = load_image("data/colorbar.png");
    image gray = rgb_to_grayscale(im);
    image gt = load_image("figs/gray.png");
    TEST(same_image(gray, gt, EPS));
    free_image(im);
    free_image(gray);
    free_image(gt);
}

void test_copy()
{
    image im = load_image("data/dog.jpg");
    image c = copy_image(im);

    image gt = load_image("data/dog.jpg");
    TEST(same_image(c, gt, EPS));
    free_image(gt);
    free_image(c);
    free_image(im);
}

void test_clamp()
{
    image im = load_image("data/dog.jpg");
    image c = copy_image(im);
    set_pixel(im, 0, 5, 10, -1);
    set_pixel(im, 1, 15, 15, 1.001);
    set_pixel(im, 2, 105, 130, -0.01);
    set_pixel(im, im.c-1, im.h-1, im.w-1, -.01);

    set_pixel(c, 0, 5, 10, 0);
    set_pixel(c, 1, 15, 15, 1);
    set_pixel(c, 2, 105, 130, 0);
    set_pixel(c, im.c-1, im.h-1, im.w-1, 0);
    clamp_image(im);
    TEST(same_image(c, im, EPS));
    free_image(im);
    free_image(c);
}

void test_shift()
{
    image im = load_image("data/dog.jpg");
    image c = copy_image(im);
    shift_image(c, 1, .1);
    TEST(within_eps(c.data[0], im.data[0], EPS));
    TEST(within_eps(c.data[im.w*im.h + 13], im.data[im.w*im.h+13] + .1, EPS));
    TEST(within_eps(c.data[2*im.w*im.h + 72], im.data[2*im.w*im.h+72], EPS));
    TEST(within_eps(c.data[im.w*im.h + 47], im.data[im.w*im.h+47] + .1, EPS));
    free_image(im);
    free_image(c);
}

void test_rgb_to_hsv()
{
    image im = load_image("data/dog.jpg");
    rgb_to_hsv(im);
    image hsv = load_image("figs/dog.hsv.png");
    TEST(same_image(im, hsv, EPS));
    free_image(im);
    free_image(hsv);
}

void test_hsv_to_rgb()
{
    image im = load_image("data/dog.jpg");
    image c = copy_image(im);
    rgb_to_hsv(im);
    hsv_to_rgb(im);
    TEST(same_image(im, c, EPS));
    free_image(im);
    free_image(c);
}

void test_nn_interpolate()
{
    image im = load_image("data/dogsmall.jpg");
    TEST(within_eps(nn_interpolate(im, 0, -.5, -.5)  , 0.231373, EPS));
    TEST(within_eps(nn_interpolate(im, 1, .5, -.5)   , 0.239216, EPS));
    TEST(within_eps(nn_interpolate(im, 2, .5, .499)  , 0.207843, EPS));
    TEST(within_eps(nn_interpolate(im, 1, 15.9, 14.2), 0.690196, EPS));
    free_image(im);
}

void test_bl_interpolate()
{
    image im = load_image("data/dogsmall.jpg");
    TEST(within_eps(bilinear_interpolate(im, 0, -.5, -.5)  , 0.231373, EPS));
    TEST(within_eps(bilinear_interpolate(im, 1, .5, -.5)   , 0.237255, EPS));
    TEST(within_eps(bilinear_interpolate(im, 2, .5, .499)  , 0.206861, EPS));
    TEST(within_eps(bilinear_interpolate(im, 1, 15.9, 14.2), 0.678588, EPS));
    free_image(im);
}



void test_nn_resize()
{
    image im = load_image("data/dogsmall.jpg");
    image resized = nn_resize(im, im.h*4, im.w*4);
    image gt = load_image("figs/dog4x-nn-for-test.png");
    TEST(same_image(resized, gt, EPS));
    free_image(im);
    free_image(resized);
    free_image(gt);

    image im2 = load_image("data/dog.jpg");
    image resized2 = nn_resize(im2, 467, 713);
    image gt2 = load_image("figs/dog-resize-nn.png");
    TEST(same_image(resized2, gt2, EPS));
    free_image(im2);
    free_image(resized2);
    free_image(gt2);
}

void test_bl_resize()
{
    image im = load_image("data/dogsmall.jpg");
    image resized = bilinear_resize(im, im.h*4, im.w*4);
    image gt = load_image("figs/dog4x-bl.png");
    TEST(same_image(resized, gt, EPS));
    free_image(im);
    free_image(resized);
    free_image(gt);

    image im2 = load_image("data/dog.jpg");
    image resized2 = bilinear_resize(im2, 467, 713);
    image gt2 = load_image("figs/dog-resize-bil.png");
    TEST(same_image(resized2, gt2, EPS));
    free_image(im2);
    free_image(resized2);
    free_image(gt2);
}

void test_multiple_resize()
{
    image im = load_image("data/dog.jpg");
    int i;
    for (i = 0; i < 10; i++){
        image im1 = bilinear_resize(im, im.h*4, im.w*4);
        image im2 = bilinear_resize(im1, im1.h/4, im1.w/4);
        free_image(im);
        free_image(im1);
        im = im2;
    }
    image gt = load_image("figs/dog-multipleresize.png");
    TEST(same_image(im, gt, EPS));
    free_image(im);
    free_image(gt);
}


void test_highpass_filter(){
    image im = load_image("data/dog.jpg");
    image f = make_highpass_filter();
    image blur = convolve_image(im, f, 0);
    clamp_image(blur);


    image gt = load_image("figs/dog-highpass.png");
    TEST(same_image(blur, gt, EPS));
    free_image(im);
    free_image(f);
    free_image(blur);
    free_image(gt);
}

void test_emboss_filter(){
    image im = load_image("data/dog.jpg");
    image f = make_emboss_filter();
    image blur = convolve_image(im, f, 1);
    clamp_image(blur);


    image gt = load_image("figs/dog-emboss.png");
    TEST(same_image(blur, gt, EPS));
    free_image(im);
    free_image(f);
    free_image(blur);
    free_image(gt);
}

void test_sharpen_filter(){
    image im = load_image("data/dog.jpg");
    image f = make_sharpen_filter();
    image blur = convolve_image(im, f, 1);
    clamp_image(blur);


    image gt = load_image("figs/dog-sharpen.png");
    TEST(same_image(blur, gt, EPS));
    free_image(im);
    free_image(f);
    free_image(blur);
    free_image(gt);
}

void test_convolution(){
    image im = load_image("data/dog.jpg");
    image f = make_box_filter(7);
    image blur = convolve_image(im, f, 1);
    clamp_image(blur);

    image gt = load_image("figs/dog-box7.png");
    TEST(same_image(blur, gt, EPS));
    free_image(im);
    free_image(f);
    free_image(blur);
    free_image(gt);
}

void test_gaussian_filter(){
    image f = make_gaussian_filter(7);
    int i;

    for(i = 0; i < f.w * f.h * f.c; i++){
        f.data[i] *= 100;
    }

    image gt = load_image("figs/gaussian_filter_7.png");
    TEST(same_image(f, gt, EPS));
    free_image(f);
    free_image(gt);
}

void test_gaussian_blur(){
    image im = load_image("data/dog.jpg");
    image f = make_gaussian_filter(2);
    image blur = convolve_image(im, f, 1);
    clamp_image(blur);

    image gt = load_image("figs/dog-gauss2.png");
    TEST(same_image(blur, gt, EPS));
    free_image(im);
    free_image(f);
    free_image(blur);
    free_image(gt);
}

void test_hybrid_image(){
    image melisa = load_image("data/melisa.png");
    image aria = load_image("data/aria.png");
    image f = make_gaussian_filter(2);
    image lfreq_m = convolve_image(melisa, f, 1);
    image lfreq_a = convolve_image(aria, f, 1);
    image hfreq_a = sub_image(aria , lfreq_a);
    image reconstruct = add_image(lfreq_m , hfreq_a);
    image gt = load_image("figs/hybrid.png");
    clamp_image(reconstruct);
    TEST(same_image(reconstruct, gt, EPS));
    free_image(melisa);
    free_image(aria);
    free_image(f);
    free_image(lfreq_m);
    free_image(lfreq_a);
    free_image(hfreq_a);
    free_image(reconstruct);
    free_image(gt);
}

void test_frequency_image(){
    image im = load_image("data/dog.jpg");
    image f = make_gaussian_filter(2);
    image lfreq = convolve_image(im, f, 1);
    image hfreq = sub_image(im, lfreq);
    image reconstruct = add_image(lfreq , hfreq);

    image low_freq = load_image("figs/low-frequency.png");
    image high_freq = load_image("figs/high-frequency-clamp.png");

    clamp_image(lfreq);
    clamp_image(hfreq);
    TEST(same_image(lfreq, low_freq, EPS));
    TEST(same_image(hfreq, high_freq, EPS));
    TEST(same_image(reconstruct, im, EPS));
    free_image(im);
    free_image(f);
    free_image(lfreq);
    free_image(hfreq);
    free_image(reconstruct);
    free_image(low_freq);
    free_image(high_freq);
}

void test_sobel(){
    image im = load_image("data/dog.jpg");
    image *res = sobel_image(im);
    image mag = res[0];
    image theta = res[1];
    feature_normalize2(mag);
    feature_normalize2(theta);

    image gt_mag = load_image("figs/magnitude.png");
    image gt_theta = load_image("figs/theta.png");
    TEST(gt_mag.w == mag.w && gt_theta.w == theta.w);
    TEST(gt_mag.h == mag.h && gt_theta.h == theta.h);
    TEST(gt_mag.c == mag.c && gt_theta.c == theta.c);
    if( gt_mag.w != mag.w || gt_theta.w != theta.w || 
            gt_mag.h != mag.h || gt_theta.h != theta.h || 
            gt_mag.c != mag.c || gt_theta.c != theta.c ) return;
    int i;
    for(i = 0; i < gt_mag.w*gt_mag.h; ++i){
        if(within_eps(gt_mag.data[i], 0, EPS)){
            gt_theta.data[i] = 0;
            theta.data[i] = 0;
        }
        if(within_eps(gt_theta.data[i], 0, EPS) || within_eps(gt_theta.data[i], 1, EPS)){
            gt_theta.data[i] = 0;
            theta.data[i] = 0;
        }
    }

    TEST(same_image(mag, gt_mag, EPS));
    TEST(same_image(theta, gt_theta, EPS));
    free_image(im);
    free_image(mag);
    free_image(theta);
    free_image(gt_mag);
    free_image(gt_theta);
    free(res);
}

void test_structure()
{
    image im = load_image("data/dogbw.png");
    image s = structure_matrix(im, 2);
    feature_normalize2(s);
    image gt = load_image("figs/structure.png");
    TEST(same_image(s, gt, EPS));
    free_image(im);
    free_image(s);
    free_image(gt);
}

void test_cornerness()
{
    image im = load_image("data/dogbw.png");
    image s = structure_matrix(im, 2);
    image c = cornerness_response(s);
    feature_normalize2(c);
    image gt = load_image("figs/response.png");
    TEST(same_image(c, gt, EPS));
    free_image(im);
    free_image(s);
    free_image(c);
    free_image(gt);
}



void test_projection()
{
    matrix H = make_translation_homography(12.4, -3.2);
    TEST(same_point(project_point(H, make_point(0,0)), make_point(12.4, -3.2), EPS));
    free_matrix(H);

    H = make_identity_homography();
    H.data[0][0] = 1.32;
    H.data[0][1] = -1.12;
    H.data[0][2] = 2.52;
    H.data[1][0] = -.32;
    H.data[1][1] = -1.2;
    H.data[1][2] = .52;
    H.data[2][0] = -3.32;
    H.data[2][1] = 1.87;
    H.data[2][2] = .112;
    point p = project_point(H, make_point(3.14, 1.59));
    TEST(same_point(p, make_point(-0.66544, 0.326017), EPS));
    free_matrix(H);
}

void test_compute_homography()
{
    match *m = calloc(4, sizeof(match));
    m[0].p = make_point(0,0);
    m[0].q = make_point(10,10);
    m[1].p = make_point(3,3);
    m[1].q = make_point(13,13);
    m[2].p = make_point(-1.2,-3.4);
    m[2].q = make_point(8.8,6.6);
    m[3].p = make_point(9,10);
    m[3].q = make_point(19,20);
    matrix H = compute_homography(m, 4);
    matrix d10 = make_translation_homography(10, 10);
    TEST(same_matrix(H, d10));
    free_matrix(H);
    free_matrix(d10);

    m[0].p = make_point(7.2,1.3);
    m[0].q = make_point(10,10.9);
    m[1].p = make_point(3,3);
    m[1].q = make_point(1.3,7.3);
    m[2].p = make_point(-.2,-3.4);
    m[2].q = make_point(.8,2.6);
    m[3].p = make_point(-3.2,2.4);
    m[3].q = make_point(1.5,-4.2);
    H = compute_homography(m, 4);
    matrix Hp = make_identity_homography();
    Hp.data[0][0] = -0.1328042; Hp.data[0][1] = -0.2910411; Hp.data[0][2] = 0.8103200;
    Hp.data[1][0] = -0.0487439; Hp.data[1][1] = -1.3077799; Hp.data[1][2] = 1.4796660;
    Hp.data[2][0] = -0.0788730; Hp.data[2][1] = -0.3727209; Hp.data[2][2] = 1.0000000;
    TEST(same_matrix(H, Hp));
    free_matrix(H);
    free_matrix(Hp);
    free(m);
}

void test_activate_matrix()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix truth_alog = load_matrix("data/test/alog.matrix");
    matrix truth_arelu = load_matrix("data/test/arelu.matrix");
    matrix truth_alrelu = load_matrix("data/test/alrelu.matrix");
    matrix truth_asoft = load_matrix("data/test/asoft.matrix");
    matrix alog = copy_matrix(a);
    activate_matrix(alog, LOGISTIC);
    matrix arelu = copy_matrix(a);
    activate_matrix(arelu, RELU);
    matrix alrelu = copy_matrix(a);
    activate_matrix(alrelu, LRELU);
    matrix asoft = copy_matrix(a);
    activate_matrix(asoft, SOFTMAX);
    TEST(same_matrix(truth_alog, alog));
    TEST(same_matrix(truth_arelu, arelu));
    TEST(same_matrix(truth_alrelu, alrelu));
    TEST(same_matrix(truth_asoft, asoft));
    free_matrix(a);
    free_matrix(alog);
    free_matrix(arelu);
    free_matrix(alrelu);
    free_matrix(asoft);
    free_matrix(truth_alog);
    free_matrix(truth_arelu);
    free_matrix(truth_alrelu);
    free_matrix(truth_asoft);
}

void test_gradient_matrix()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix y = load_matrix("data/test/y.matrix");
    matrix truth_glog = load_matrix("data/test/glog.matrix");
    matrix truth_grelu = load_matrix("data/test/grelu.matrix");
    matrix truth_glrelu = load_matrix("data/test/glrelu.matrix");
    matrix truth_gsoft = load_matrix("data/test/gsoft.matrix");
    matrix glog = copy_matrix(a);
    matrix grelu = copy_matrix(a);
    matrix glrelu = copy_matrix(a);
    matrix gsoft = copy_matrix(a);
    gradient_matrix(y, LOGISTIC, glog);
    gradient_matrix(y, RELU, grelu);
    gradient_matrix(y, LRELU, glrelu);
    gradient_matrix(y, SOFTMAX, gsoft);
    TEST(same_matrix(truth_glog, glog));
    TEST(same_matrix(truth_grelu, grelu));
    TEST(same_matrix(truth_glrelu, glrelu));
    TEST(same_matrix(truth_gsoft, gsoft));
    free_matrix(a);
    free_matrix(y);
    free_matrix(glog);
    free_matrix(grelu);
    free_matrix(glrelu);
    free_matrix(gsoft);
    free_matrix(truth_glog);
    free_matrix(truth_grelu);
    free_matrix(truth_glrelu);
    free_matrix(truth_gsoft);
}

void test_layer()
{
    matrix a = load_matrix("data/test/a.matrix");
    matrix w = load_matrix("data/test/w.matrix");
    matrix dw = load_matrix("data/test/dw.matrix");
    matrix v = load_matrix("data/test/v.matrix");
    matrix delta = load_matrix("data/test/delta.matrix");

    matrix truth_dx = load_matrix("data/test/truth_dx.matrix");
    matrix truth_v = load_matrix("data/test/truth_v.matrix");
    matrix truth_dw = load_matrix("data/test/truth_dw.matrix");

    matrix updated_dw = load_matrix("data/test/updated_dw.matrix");
    matrix updated_w = load_matrix("data/test/updated_w.matrix");
    matrix updated_v = load_matrix("data/test/updated_v.matrix");

    matrix truth_out = load_matrix("data/test/out.matrix");
    layer l = make_layer(64, 16, LRELU);
    free_matrix(l.w);
    free_matrix(l.dw);
    free_matrix(l.v);
    l.w = w;
    l.dw = dw;
    l.v = v;
    matrix out = forward_layer(&l, a);
    TEST(same_matrix(truth_out, out));

    matrix dx = backward_layer(&l, delta);
    TEST(same_matrix(truth_v, v));
    TEST(same_matrix(truth_dw, l.dw));
    TEST(same_matrix(truth_dx, dx));

    update_layer(&l, .01, .9, .01);
    TEST(same_matrix(updated_dw, l.dw));
    TEST(same_matrix(updated_w, l.w));
    TEST(same_matrix(updated_v, l.v));
    //free_layer(l);
    free_matrix(l.w);
    free_matrix(l.dw);
    free_matrix(l.v);
    free_matrix(l.in);
    free_matrix(l.out);
    //free_matrix(a);
    //free_matrix(w);
    //free_matrix(v);
    free_matrix(delta);
    free_matrix(truth_dx);
    free_matrix(truth_v);
    free_matrix(truth_dw);
    free_matrix(updated_dw);
    free_matrix(updated_w);
    free_matrix(updated_v);
    free_matrix(truth_out);
    //free_matrix(out);
    free_matrix(dx);
}

void make_matrix_test()
{
    srand(1);
    matrix a = random_matrix(32, 64, 10);
    matrix w = random_matrix(64, 16, 10);
    matrix y = random_matrix(32, 64, 10);
    matrix dw = random_matrix(64, 16, 10);
    matrix v = random_matrix(64, 16, 10);
    matrix delta = random_matrix(32, 16, 10);

    save_matrix(a, "data/test/a.matrix");
    save_matrix(w, "data/test/w.matrix");
    save_matrix(dw, "data/test/dw.matrix");
    save_matrix(v, "data/test/v.matrix");
    save_matrix(delta, "data/test/delta.matrix");
    save_matrix(y, "data/test/y.matrix");

    matrix alog = copy_matrix(a);
    activate_matrix(alog, LOGISTIC);
    save_matrix(alog, "data/test/alog.matrix");

    matrix arelu = copy_matrix(a);
    activate_matrix(arelu, RELU);
    save_matrix(arelu, "data/test/arelu.matrix");

    matrix alrelu = copy_matrix(a);
    activate_matrix(alrelu, LRELU);
    save_matrix(alrelu, "data/test/alrelu.matrix");

    matrix asoft = copy_matrix(a);
    activate_matrix(asoft, SOFTMAX);
    save_matrix(asoft, "data/test/asoft.matrix");


    matrix glog = copy_matrix(a);
    gradient_matrix(y, LOGISTIC, glog);
    save_matrix(glog, "data/test/glog.matrix");

    matrix grelu = copy_matrix(a);
    gradient_matrix(y, RELU, grelu);
    save_matrix(grelu, "data/test/grelu.matrix");

    matrix glrelu = copy_matrix(a);
    gradient_matrix(y, LRELU, glrelu);
    save_matrix(glrelu, "data/test/glrelu.matrix");

    matrix gsoft = copy_matrix(a);
    gradient_matrix(y, SOFTMAX, gsoft);
    save_matrix(gsoft, "data/test/gsoft.matrix");


    layer l = make_layer(64, 16, LRELU);
    l.w = w;
    l.dw = dw;
    l.v = v;

    matrix out = forward_layer(&l, a);
    save_matrix(out, "data/test/out.matrix");

    matrix dx = backward_layer(&l, delta);
    save_matrix(l.dw, "data/test/truth_dw.matrix");
    save_matrix(l.v, "data/test/truth_v.matrix");
    save_matrix(dx, "data/test/truth_dx.matrix");

    update_layer(&l, .01, .9, .01);
    save_matrix(l.dw, "data/test/updated_dw.matrix");
    save_matrix(l.w, "data/test/updated_w.matrix");
    save_matrix(l.v, "data/test/updated_v.matrix");
}

void test_hw0()
{
    test_get_pixel();
    test_set_pixel();
    test_copy();
    test_shift();
    test_clamp();
    test_grayscale();
    test_rgb_to_hsv();
    test_hsv_to_rgb();
    printf("%d tests, %d passed, %d failed\n", tests_total, tests_total-tests_fail, tests_fail);
}
void test_hw1()
{
    test_nn_interpolate();
    test_nn_resize();
    test_bl_interpolate();
    test_bl_resize();
    test_multiple_resize();
    printf("%d tests, %d passed, %d failed\n", tests_total, tests_total-tests_fail, tests_fail);
}
void test_hw2()
{
    test_gaussian_filter();
    test_sharpen_filter();
    test_emboss_filter();
    test_highpass_filter();
    test_convolution();
    test_gaussian_blur();
    test_hybrid_image();
    test_frequency_image();
    test_sobel();
    printf("%d tests, %d passed, %d failed\n", tests_total, tests_total-tests_fail, tests_fail);
}
void test_hw3()
{
    test_structure();
    test_cornerness();
    test_projection();
    test_compute_homography();
    printf("%d tests, %d passed, %d failed\n", tests_total, tests_total-tests_fail, tests_fail);
}
void make_hw4_tests()
{
    image dots = load_image("data/dots.png");
    image intdot = make_integral_image(dots);
    save_image_binary(intdot, "data/dotsintegral.bin");

    image dogbw = load_image("data/dogbw.png");
    image intdog = make_integral_image(dogbw);
    save_image_binary(intdog, "data/dogintegral.bin");

    image dog = load_image("data/dog.jpg");
    image smooth = box_filter_image(dog, 15);
    save_png(smooth, "data/dogbox");

    image smooth_c = center_crop(smooth);
    save_png(smooth_c, "data/dogboxcenter");

    image doga = load_image("data/dog_a_small.jpg");
    image dogb = load_image("data/dog_b_small.jpg");
    image structure = time_structure_matrix(dogb, doga, 15);
    save_image_binary(structure, "data/structure.bin");

    image velocity = velocity_image(structure, 5);
    save_image_binary(velocity, "data/velocity.bin");
}
void test_integral_image()
{
    image dots = load_image("data/dots.png");
    image intdot = make_integral_image(dots);
    image intdot_t = load_image_binary("data/dotsintegral.bin");
    TEST(same_image(intdot, intdot_t, EPS));

    image dog = load_image("data/dogbw.png");
    image intdog = make_integral_image(dog);
    image intdog_t = load_image_binary("data/dogintegral.bin");
    TEST(same_image(intdog, intdog_t, .6));

    free_image(dots);
    free_image(intdot);
    free_image(intdot_t);
    free_image(dog);
    free_image(intdog);
    free_image(intdog_t);
}
void test_exact_box_filter_image()
{
    image dog = load_image("data/dog.jpg");
    image smooth = box_filter_image(dog, 15);
    image smooth_t = load_image("data/dogbox.png");
    //printf("avg origin difference test: %f\n", avg_diff(smooth, dog));
    //printf("avg smooth difference test: %f\n", avg_diff(smooth, smooth_t));
    TEST(same_image(smooth, smooth_t, EPS*2));

    free_image(dog);
    free_image(smooth);
    free_image(smooth_t);
}

void test_good_enough_box_filter_image()
{
    image dog = load_image("data/dog.jpg");
    image smooth = box_filter_image(dog, 15);
    image smooth_c = center_crop(smooth);
    image smooth_t = load_image("data/dogboxcenter.png");
    image dog_c = center_crop(dog);
    printf("avg origin difference test: %f\n", avg_diff(smooth_c, dog_c));
    printf("avg smooth difference test: %f\n", avg_diff(smooth_c, smooth_t));
    TEST(same_image(smooth_c, smooth_t, EPS*2));

    free_image(dog);
    free_image(dog_c);
    free_image(smooth);
    free_image(smooth_t);
    free_image(smooth_c);
}
void test_structure_image()
{
    image doga = load_image("data/dog_a_small.jpg");
    image dogb = load_image("data/dog_b_small.jpg");
    image structure = time_structure_matrix(dogb, doga, 15);
    image structure_t = load_image_binary("data/structure.bin");
    image structure_c = center_crop(structure);
    image structure_tc = center_crop(structure_t);
    TEST(same_image(structure_c, structure_tc, EPS));

    free_image(doga);
    free_image(dogb);
    free_image(structure);
    free_image(structure_t);
    free_image(structure_c);
    free_image(structure_tc);
}
void test_velocity_image()
{
    image structure = load_image_binary("data/structure.bin");
    image velocity = velocity_image(structure, 5);
    image velocity_t = load_image_binary("data/velocity.bin");
    TEST(same_image(velocity, velocity_t, EPS));
    free_image(structure);
    free_image(velocity);
    free_image(velocity_t);
}
void test_hw4()
{
    test_integral_image();
    test_exact_box_filter_image();
    test_good_enough_box_filter_image();
    test_structure_image();
    test_velocity_image();
    printf("%d tests, %d passed, %d failed\n", tests_total, tests_total-tests_fail, tests_fail);
}
void test_hw5()
{
    test_activate_matrix();
    test_gradient_matrix();
    test_layer();
    printf("%d tests, %d passed, %d failed\n", tests_total, tests_total-tests_fail, tests_fail);
}

void run_tests()
{
    test_structure();
    test_cornerness();
    printf("%d tests, %d passed, %d failed\n", tests_total, tests_total-tests_fail, tests_fail);
}

