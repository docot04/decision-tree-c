#include <stdio.h>
#include <stdlib.h>
#include "data/dataset.h"
#include "model/decision_tree.h"
int main() {
    Dataset *dataset = loadCSV("dataset.csv");
    if (!dataset) return 1;
    int targetIndex = dataset->cols - 1;
    int *rows = malloc(dataset->rows * sizeof(int));
    for (int i = 0; i < dataset->rows; i++) rows[i] = i;
    int featureCount = targetIndex;
    int *usedFeatures = calloc(featureCount, sizeof(int));

    // train the tree
    TreeNode *root = buildTree(dataset, rows, dataset->rows, targetIndex, usedFeatures);
    printf("Decision tree trained!\n");
    printf("\nTree:\n");
    printTree(root, dataset, 0);
    free(rows);
    free(usedFeatures);
    freeTree(root);
    freeDataset(dataset);
    return 0;
}