#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data/dataset.h"
#include "train/decision_tree.h"
#include "test/prediction.h"
#include "model/model.h"


// train a decision tree and save it to a ".dt" file
int trainModel(const char *datasetFile, const char *modelFile) {
    printf("Loading dataset: %s\n", datasetFile);
    Dataset *dataset = loadCSV(datasetFile);
    if (!dataset) {
        printf("Failed to load dataset\n");
        return 1;
    }

    // last col is target
    int targetIndex = dataset->cols - 1;
    int *rows = malloc(dataset->rows * sizeof(int));
    if (!rows) {
        printf("Failed to allocate rows\n");
        freeDataset(dataset);
        return 1;
    }
    for (int i = 0; i < dataset->rows; i++) rows[i] = i;
    
    // track already used features
    int featureCount = targetIndex;
    int *usedFeatures = calloc(featureCount, sizeof(int));
    if (!usedFeatures) {
        printf("Failed to allocate feature tracking\n");
        free(rows);
        freeDataset(dataset);
        return 1;
    }

    // build the decision tree
    printf("Training decision tree...\n");
    TreeNode *root = buildTree(dataset, rows, dataset->rows, targetIndex, usedFeatures);
    if (!root) {
        printf("Failed to build decision tree\n");
        free(rows);
        free(usedFeatures);
        freeDataset(dataset);
        return 1;
    }
    printf("Decision tree trained!\n");

    // DEBUG: DISPLAY THE TREE
    printf("\nTree:\n");
    printTree(root, dataset,0);

    // save the trained model
    printf("\nSaving model: %s\n", modelFile);
    if (!saveModel(modelFile, root, dataset)) {
        printf("Failed to save model\n");
        free(rows);
        free(usedFeatures);
        freeTree(root);
        freeDataset(dataset);
        return 1;
    }

    printf("Model saved successfully!\n");
    free(rows);
    free(usedFeatures);
    freeTree(root);
    freeDataset(dataset);
    return 0;
}


// load a trained model from ".dt" file and make a prediction
int testModel(const char *modelFile, const char *input) {
    printf("Loading model: %s\n", modelFile);
    Dataset *dataset = NULL;
    TreeNode *root = loadModel(modelFile, &dataset);
    if (!root) {
        printf("Failed to load model\n");
        return 1;
    }

    char *inputCopy = malloc(strlen(input) + 1);
    if (!inputCopy) {
        printf("Failed to allocate input buffer\n");
        freeTree(root);
        freeDataset(dataset);
        return 1;
    }
    strcpy(inputCopy, input);

    int featureCount = dataset->cols - 1;
    char **values = malloc(featureCount * sizeof(char *));
    if (!values) {
        printf("Failed to allocate input values\n");
        free(inputCopy);
        freeTree(root);
        freeDataset(dataset);
        return 1;
    }

    char *token = strtok(inputCopy, ",");
    int count = 0;
    while (token != NULL && count < featureCount) {
        values[count] = token;
        count++;
        token = strtok(NULL, ",");
    }

    if (count != featureCount || token != NULL) {
        printf("Invalid input: expected %d features\n", featureCount);
        free(values);
        free(inputCopy);
        freeTree(root);
        freeDataset(dataset);
        return 1;
    }

    // encode string into IDS
    int *sample = malloc(featureCount * sizeof(int));
    if (!sample) {
        printf("Failed to allocate sample\n");
        free(values);
        free(inputCopy);
        freeTree(root);
        freeDataset(dataset);
        return 1;
    }


    if (!encodeSample(dataset, values, sample)) {
        printf("Invalid feature value\n");
        free(sample);
        free(values);
        free(inputCopy);
        freeTree(root);
        freeDataset(dataset);
        return 1;
    }

    // display encoded sample
    printf("\nEncoded sample: ");
    for (int i = 0; i < featureCount; i++) printf("%d ", sample[i]);
    printf("\n");

    // run prediction
    int prediction = predict(root, sample);

    // convert predicted class ID into string
    int targetIndex = dataset->cols - 1;
    printf("Prediction: %d\n", prediction);
    printf("Prediction label: %s\n", dataset->valueNames[targetIndex][prediction]);

    free(sample);
    free(values);
    free(inputCopy);
    freeTree(root);
    freeDataset(dataset);
    return 0;
}


int main(int argc, char *argv[]) {
    
    if (argc < 2) {
        printf("Usage:\n");
        printf("  %s train <dataset.csv> <model.dt>\n", argv[0]);
        printf("  %s predict <model.dt> \"value1,value2,...\"\n", argv[0]);
        return 1;
    }

    // dt train dataset.csv model.dt
    if (strcmp(argv[1], "train") == 0) {
        if (argc != 4) {
            printf("Usage: %s train <dataset.csv> <model.dt>\n", argv[0]);
            return 1;
        }
        return trainModel(argv[2], argv[3]);
    }

    // dt predict model.dt "Rain,Cool,Normal,False"
    if (strcmp(argv[1], "predict") == 0) {
        if (argc != 4) {
            printf("Usage: %s predict <model.dt> \"value1,value2,...\"\n", argv[0]);
            return 1;
        }
        return testModel(argv[2], argv[3]);
    }

    // unknown
    printf("Unknown command: %s\n", argv[1]);
    return 1;
}