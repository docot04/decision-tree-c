#ifndef MODEL_H
#define MODEL_H
#include "../data/dataset.h"
#include "../train/decision_tree.h"

// convert trained tree into a custom ".dt" file
int saveModel(const char *filename, TreeNode *root, Dataset *dataset);

// load trained tree from ".dt" file
TreeNode *loadModel(const char *filename, Dataset **dataset);

#endif