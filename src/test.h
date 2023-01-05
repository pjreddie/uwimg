#ifndef TEST_H
#define TEST_H
#include <stdio.h>
#define EPS .002
extern int tests_total;
extern int tests_fail;
#define TEST(EX) do { ++tests_total; if(!(EX)) {\
    fprintf(stderr, "failed: [%s] testing [%s] in %s, line %d\n", __FUNCTION__, #EX, __FILE__, __LINE__); \
    ++tests_fail; }else{fprintf(stderr, "passed: [%s] testing [%s] in %s, line %d\n", __FUNCTION__, #EX, __FILE__, __LINE__);}} while (0)

void run_tests();
void test_hw0();
void test_hw1();
void test_hw2();
void test_hw3();
void test_hw4();
void test_hw5();
#endif
