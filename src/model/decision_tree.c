#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "decision_tree.h"

double entropy(int *labels, int count) {
    /**
     * H(S) = -Summation(i=1 to n) { p[i] * log(base2){ p[i] } }
     * where
     * p[i] : probability of class i
     * n    : number of classes
     */
    if (count == 0) return 0.0;
    int class0 = 0;
    int class1 = 0;
    for (int i = 0; i < count; i++) {
        if (labels[i] == 0) class0++;
        else if (labels[i] == 1) class1++;
    }
    double p0 = (double)class0 / count;
    double p1 = (double)class1 / count;
    double result = 0.0;
    if (p0 > 0) result -= p0 * log2(p0);
    if (p1 > 0) result -= p1 * log2(p1);
    return result;
}

double informationGain(Dataset *dataset, int *rows, int rowCount, int featureIndex, int targetIndex) {
    /**
     * IG(S,A) = H(S) - Summation(v belongs to Values(A)) { mod(S[v]) / mod(S) * H(S[v]) }
     * i.e.
     * IG = Entropy before split - Entropy after split
     */

    // entropy of entire input dataset
    int *labels = malloc(rowCount * sizeof(int));
    for (int i = 0; i < rowCount; i++) labels[i] = dataset->data[rows[i]][targetIndex];
    double parentEntropy = entropy(labels, rowCount);
    free(labels);
    
    // calculate weighted entropy after splitting
    double weightedEntropy = 0.0;
    int numberOfValues = dataset->numValues[featureIndex];
    for (int value = 0; value < numberOfValues; value++) {
        int subsetSize = 0;
        for (int i = 0; i < rowCount; i++) {
            int row = rows[i];
            if (dataset->data[row][featureIndex] == value) subsetSize++;
        }
        if (subsetSize == 0) continue;
        int *subsetLabels = malloc(subsetSize * sizeof(int));
        int index = 0;
        for (int i = 0; i < rowCount; i++) {
            int row = rows[i];
            if (dataset->data[row][featureIndex] == value) {
                subsetLabels[index] = dataset->data[row][targetIndex];
                index++;
            }
        }
        double subsetEntropy = entropy(subsetLabels, subsetSize);
        double weight = (double)subsetSize / rowCount;
        weightedEntropy += weight * subsetEntropy;
        free(subsetLabels);
    }
    return parentEntropy - weightedEntropy;
}

int majorityClass(Dataset *dataset, int *rows, int rowCount, int targetIndex) {
    int count0 = 0;
    int count1 = 0;
    for (int i = 0; i < rowCount; i++) {
        int label = dataset->data[rows[i]][targetIndex];
        if (label == 0) count0++;
        else if (label == 1) count1++;
    }
    if (count1 > count0) return 1;
    return 0;
}

TreeNode *buildTree(Dataset *dataset, int *rows, int rowCount, int targetIndex, int *usedFeatures) {

    // check whether all rows have same class
    int firstClass = dataset->data[rows[0]][targetIndex];
    int allSame = 1;
    for (int i = 1; i < rowCount; i++) {
        if (dataset->data[rows[i]][targetIndex] != firstClass) {
            allSame = 0;
            break;
        }
    }

    // if all rows have same class create leaf
    if (allSame) {
        TreeNode *node = malloc(sizeof(TreeNode));
        node->isLeaf = 1;
        node->classLabel = firstClass;
        node->featureIndex = -1;
        node->numChildren = 0;
        node->featureValues = NULL;
        node->children = NULL;
        return node;
    }

    // find feature with highest IG
    int bestFeature = -1;
    double bestGain = -1.0;
    for (int feature = 0; feature < targetIndex; feature++) {
        if (usedFeatures[feature]) continue;
        double gain = informationGain(dataset, rows, rowCount, feature, targetIndex);
        if (gain > bestGain) {
            bestGain = gain;
            bestFeature = feature;
        }
    }

    // if no features remain, crete a leaf with majority class
    if (bestFeature == -1) {
        TreeNode *node = malloc(sizeof(TreeNode));
        node->isLeaf = 1;
        node->classLabel = majorityClass(dataset, rows, rowCount, targetIndex);
        node->featureIndex = -1;
        node->numChildren = 0;
        node->featureValues = NULL;
        node->children = NULL;
        return node;
    }

    // create an internal node
    TreeNode *node = malloc(sizeof(TreeNode));
    node->isLeaf = 0;
    node->featureIndex = bestFeature;
    node->classLabel = -1;
    node->numChildren = dataset->numValues[bestFeature];
    node->featureValues = malloc(node->numChildren * sizeof(int));
    node->children = malloc(node->numChildren *sizeof(TreeNode *));

    // mark the feature as used so that it isn't used again
    usedFeatures[bestFeature] = 1;

    // build each branch recursively
    for (int value = 0; value < node->numChildren; value++) {
        node->featureValues[value] = value;
        int subsetSize = 0;
        for (int i = 0; i < rowCount; i++) {
            int row = rows[i];
            if (dataset->data[row][bestFeature] == value) subsetSize++;
        }

        if (subsetSize == 0) {
            node->children[value] = NULL;
            continue;
        }

        int *subsetRows = malloc(subsetSize *sizeof(int));
        int index = 0;
        for (int i = 0; i < rowCount; i++) {
            int row = rows[i];
            if (dataset->data[row][bestFeature] == value) {
                subsetRows[index] = row;
                index++;
            }
        }

        // make a copy of usedFeatures because different branches can use different feature
        int featureCount = targetIndex;
        int *childUsedFeatures = malloc(featureCount * sizeof(int));
        for (int i = 0; i < featureCount; i++) childUsedFeatures[i] = usedFeatures[i];

        // recursively build child
        node->children[value] = buildTree(dataset, subsetRows, subsetSize, targetIndex, childUsedFeatures);
        free(childUsedFeatures);
        free(subsetRows);
    }

    // we dont modify the parent's usedFeatures array directly bevcause every child gets its own copy
    return node;
}

// helper function
void printIndent(int depth) {
    for (int i = 0; i < depth; i++) printf("    ");
}

void printTree(TreeNode *node, Dataset *dataset, int depth ) {
    if (!node) return;

    // leaf node
    if (node->isLeaf) {
        printIndent(depth);
        printf("Predict: %d\n",node->classLabel);
        return;
    }
    
    // internal node
    printIndent(depth);
    printf("Feature: %s\n",dataset->featureNames[node->featureIndex]);
    for (int i = 0; i < node->numChildren;i++) {
        printIndent(depth);
        printf("Value: %d\n", node->featureValues[i]);
        printTree(node->children[i], dataset, depth + 1);
    }
}

void freeTree(TreeNode *node) {
    if (!node) return;

    if (!node->isLeaf) {
        for (int i = 0; i < node->numChildren; i++) freeTree(node->children[i]);
        free(node->featureValues);
        free(node->children);
    }

    free(node);
}