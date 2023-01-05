 from uwimg import *
 im = load_image("data/dog.jpg")
 f = make_box_filter(7)
 blur = convolve_image(im, f, 1)
 save_image(blur, "dog-box7")
