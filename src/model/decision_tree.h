#ifndef DECISION_TREE_H
#define DECISION_TREE_H
#include "../data/dataset.h"

typedef struct TreeNode {
    int isLeaf;
    int featureIndex;
    int classLabel;
    int numChildren;
    int *featureValues;
    struct TreeNode **children;
} TreeNode;

// calculates entropy for a feature
double entropy(int *labels, int count);

// calculates information gain for the entire dataset
double informationGain(Dataset *dataset, int *rows, int rowCount, int featureIndex, int targetIndex);

// picks the majority class
int majorityClass(Dataset *dataset, int *rows, int rowCount, int targetIndex);

// build a tree
TreeNode *buildTree(Dataset *dataset, int *rows, int rowCount, int targetIndex, int *usedFeatures);

// free the tree from memory
void freeTree(TreeNode *node);

// DEBUGGING: print the generated tree
void printTree(TreeNode *node, Dataset *dataset, int depth);

#endif