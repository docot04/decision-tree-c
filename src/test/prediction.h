#ifndef PREDICTION_H
#define PREDICTION_H
#include "../train/decision_tree.h"
#include "../data/dataset.h"

// tree traversal prediction algorithm
int predict(TreeNode *root, int *sample);

// decode/encode categorical mappings
int encodeValue(Dataset *dataset, int featureIndex, const char *value);

// decode/encode a string of samples
int encodeSample( Dataset *dataset, char **values, int *sample);

#endif