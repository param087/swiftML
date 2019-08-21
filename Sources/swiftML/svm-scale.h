//
//  svm-scale.h
//  SVM
//
//  Created by Victor A on 7/30/19.
//  Copyright © 2019 Victor A. All rights reserved.
//

#ifndef svm_scale_h
#define svm_scale_h

#include <stdio.h>
#include "svm.h"

int scaleWith(int argc,char **argv);
void output_target(double value);
void output(int index, double value);
int clean_up(FILE *fp_restore, FILE *fp, const char *msg);
#endif /* svm_scale_h */
