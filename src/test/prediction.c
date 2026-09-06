#include <string.h>
#include <stdio.h>
#include "prediction.h"

int predict(TreeNode *root, int *sample) {
    TreeNode *current = root;
    while (!current->isLeaf) {
        int feature = current->featureIndex;
        int value = sample[feature];
        current = current->children[value];
    }
    return current->classLabel;
}

int encodeValue(Dataset *dataset, int featureIndex, const char *value) {
    for (int i = 0; i < dataset->numValues[featureIndex]; i++) {
        if (strcmp(dataset->valueNames[featureIndex][i], value) == 0) return i;
    }
    return -1;
}

int encodeSample(Dataset *dataset, char **values, int *sample) {

    // skip last column (target)
    int featureCount = dataset->cols - 1;
    for (int i = 0; i < featureCount; i++) {
        int encoded = encodeValue(dataset, i, values[i]);
        if (encoded == -1) return 0;
        sample[i] = encoded;
    }
    return 1;
}