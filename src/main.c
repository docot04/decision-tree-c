#include <stdio.h>
#include <stdlib.h>
#include "data/dataset.h"
#include "train/decision_tree.h"
#include "test/prediction.h"
#include "model/model.h"

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

    if (saveModel("model.dt",root,dataset)) printf("\nModel saved\n");

    Dataset *loadedDataset = NULL;
    TreeNode *loadedRoot = loadModel("model.dt", &loadedDataset);
    if (!loadedRoot) return 1;
    printf("\nLoaded tree:\n");
    printTree(loadedRoot, loadedDataset, 0);

    printf("\nTesting prediction from loaded model:\n");

    char *values[] = {"Snow", "Cool", "Normal", "False"};
    int sample[4];
    if (!encodeSample(loadedDataset, values, sample)) {
        printf("Invalid input\n");
    } else {
        printf("Encoded sample: ");
        for (int i = 0; i < 4; i++) printf("%d ", sample[i]);
        printf("\n");
        int prediction = predict(loadedRoot,sample);
        printf("Prediction: %d\n", prediction);
        printf("Prediction label: %s\n", loadedDataset ->valueNames[4][prediction]
        );
    }
    
    free(rows);
    free(usedFeatures);
    freeTree(root);
    freeDataset(dataset);
    return 0;
}