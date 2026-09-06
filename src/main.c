#include <stdio.h>
#include <stdlib.h>
#include "data/dataset.h"
#include "train/decision_tree.h"
#include "test/prediction.h"

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

    // prediction test
    int sample1[] = {1, 0, 0, 0};
    int prediction = predict(root, sample1);
    printf("\nPrediction 1: %d\n", prediction);
    
    printf("\n");
    printDataset(dataset);
    printf("\nValue mappings:\n");
    for (int col = 0; col < dataset->cols; col++) {
        printf("%s:\n", dataset->featureNames[col]);
        for (int value = 0; value < dataset->numValues[col]; value++) {
            printf("    %d -> %s\n", value, dataset->valueNames[col][value]);
        }
    }

    printf("\nEncoded values:\n");
    printf("Rain  -> %d\n", encodeValue(dataset, 0, "Rain"));
    printf("Cool  -> %d\n", encodeValue(dataset, 1, "Cool"));
    printf("Normal -> %d\n", encodeValue(dataset, 2, "Normal"));
    printf("True  -> %d\n", encodeValue(dataset, 3, "True"));
    printf("Invalid -> %d\n", encodeValue(dataset, 0, "Snowy"));

    printf("\nPrediction 2:\n");
    char *values[] = {"Snow", "Cool", "Normal", "False"};
    int sample[dataset->cols - 1];
    if (encodeSample(dataset, values, sample)) {
        int prediction = predict(root, sample);
        printf("\nInput: Rain, Cool, Normal, False\n");
        printf("Prediction: %d\n", prediction);
    } else {
        printf("\nInvalid input!\n");
    }


    free(rows);
    free(usedFeatures);
    freeTree(root);
    freeDataset(dataset);
    return 0;
}