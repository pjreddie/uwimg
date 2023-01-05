#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "image.h"
#include "list.h"

data random_batch(data d, int n)
{
    matrix X = {0};
    matrix y = {0};
    X.shallow = y.shallow = 1;
    X.rows = y.rows = n;
    X.cols = d.X.cols;
    y.cols = d.y.cols;
    X.data = calloc(n, sizeof(double*));
    y.data = calloc(n, sizeof(double*));
    int i;
    for(i = 0; i < n; ++i){
        int ind = rand()%d.X.rows;
        X.data[i] = d.X.data[ind];
        y.data[i] = d.y.data[ind];
    }
    data c;
    c.X = X;
    c.y = y;
    return c;
}

list *get_lines(char *filename)
{
    char *path;
    FILE *file = fopen(filename, "r");
    if(!file) {
        fprintf(stderr, "Couldn't open file %s\n", filename);
        exit(0);
    }
    list *lines = make_list();
    while((path=fgetl(file))){
        list_insert(lines, path);
    }
    fclose(file);
    return lines;
}

data load_classification_data(char *images, char *label_file, int bias)
{
    list *image_list = get_lines(images);
    list *label_list = get_lines(label_file);
    int k = label_list->size;
    char **labels = (char **)list_to_array(label_list);

    int n = image_list->size;
    node *nd = image_list->front;
    int cols = 0;
    int i;
    int count = 0;
    matrix X;
    matrix y = make_matrix(n, k);
    while(nd){
        char *path = (char *)nd->val;
        image im = load_image(path);
        if (!cols) {
            cols = im.w*im.h*im.c;
            X = make_matrix(n, cols + (bias != 0));
        }
        for (i = 0; i < cols; ++i){
            X.data[count][i] = im.data[i];
        }
        if(bias) X.data[count][cols] = 1;

        for (i = 0; i < k; ++i){
            if(strstr(path, labels[i])){
                y.data[count][i] = 1;
            }
        }
        ++count;
        nd = nd->next;
    }
    free_list(image_list);
    data d;
    d.X = X;
    d.y = y;
    return d;
}


char *fgetl(FILE *fp)
{
    if(feof(fp)) return 0;
    size_t size = 512;
    char *line = malloc(size*sizeof(char));
    if(!fgets(line, size, fp)){
        free(line);
        return 0;
    }

    size_t curr = strlen(line);

    while((line[curr-1] != '\n') && !feof(fp)){
        if(curr == size-1){
            size *= 2;
            line = realloc(line, size*sizeof(char));
            if(!line) {
                fprintf(stderr, "malloc failed %ld\n", size);
                exit(0);
            }
        }
        size_t readsize = size-curr;
        if(readsize > INT_MAX) readsize = INT_MAX-1;
        fgets(&line[curr], readsize, fp);
        curr = strlen(line);
    }
    if(curr >= 2 && line[curr-2] == '\r') line[curr-2] = '\0';
    if(curr >= 1 && line[curr-1] == '\n') line[curr-1] = '\0';

    return line;
}

void free_data(data d)
{
    free_matrix(d.X);
    free_matrix(d.y);
}



