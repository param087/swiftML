//
//  svm-train.h
//  SVM
//
//  Created by Victor A on 7/29/19.
//  Copyright © 2019 Victor A. All rights reserved.
//

#ifndef svm_train_h
#define svm_train_h

#include <stdio.h>
#include "svm.h"

struct svm_parameter param;        // set by parse_command_line
struct svm_problem prob;        // set by read_problem
struct svm_model *model;
struct svm_node *x_space;
int cross_validation;
int nr_fold;

int trainOn(int argc, char **argv, char *path);
void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
void read_problem(const char *filename);
void do_cross_validation(void);

#endif /* svm_train_h */
