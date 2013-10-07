#ifndef MKFILTER_H
#define MKFILTER_H
#include "blurt.h"

#define MAXPZ	    512

void mkfilter(const char *argv[], size_t &order_out, float alpha_out[], float beta_out[], float &gamma_out);
#endif
