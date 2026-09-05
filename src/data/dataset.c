#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dataset.h"

#define MAX_LINE 1024
#define MAX_VALUE 100

static int getValueIndex(char **values, int *count, const char *value) {
    for (int i = 0; i < *count; i++) {
        if (strcmp(values[i], value) == 0) return i;
    }
    values[*count] = malloc(strlen(value) + 1);
    strcpy(values[*count], value);
    (*count)++;
    return *count - 1;
}

Dataset *loadCSV(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Could not open dataset\n");
        return NULL;
    }
    char line[MAX_LINE];
    if (!fgets(line, MAX_LINE, file)) {
        fclose(file);
        return NULL;
    }
    line[strcspn(line, "\n")] = '\0';
    int cols = 1;
    for (char *p = line; *p; p++) {
        if (*p == ',') cols++;
    }
    char **featureNames = malloc(cols * sizeof(char *));
    char *token = strtok(line, ",");
    for (int i = 0; i < cols; i++) {
        featureNames[i] = malloc(strlen(token) + 1);
        strcpy(featureNames[i], token);
        token = strtok(NULL, ",");
    }
    int capacity = 10;
    int rows = 0;
    int **data = malloc(capacity * sizeof(int *));
    char ***uniqueValues = malloc(cols * sizeof(char **));
    int *numValues = calloc(cols, sizeof(int));
    for (int i = 0; i < cols; i++) uniqueValues[i] = malloc(MAX_VALUE * sizeof(char *));
    while (fgets(line, MAX_LINE, file)) {
        line[strcspn(line, "\n")] = '\0';
        if (rows >= capacity) {
            capacity *= 2;
            data = realloc(data, capacity * sizeof(int *));
        }
        data[rows] = malloc(cols * sizeof(int));
        token = strtok(line, ",");
        for (int col = 0; col < cols; col++) {
            data[rows][col] = getValueIndex(uniqueValues[col], &numValues[col], token);
            token = strtok(NULL, ",");
        }
        rows++;
    }
    fclose(file);
    for (int col = 0; col < cols; col++) {
        for (int i = 0; i < numValues[col]; i++) free(uniqueValues[col][i]);
        free(uniqueValues[col]);
    }
    free(uniqueValues);
    Dataset *dataset = malloc(sizeof(Dataset));
    dataset->rows = rows;
    dataset->cols = cols;
    dataset->data = data;
    dataset->featureNames = featureNames;
    dataset->numValues = numValues;
    return dataset;
}

void printDataset(Dataset *dataset) {
    for (int i = 0; i < dataset->cols; i++) printf("%s\t", dataset->featureNames[i]);
    printf("\n");
    for (int row = 0; row < dataset->rows; row++) {
        for (int col = 0; col < dataset->cols; col++) printf("%d\t", dataset->data[row][col]);
        printf("\n");
    }
}

void freeDataset(Dataset *dataset) {
    if (!dataset) return;
    for (int i = 0; i < dataset->rows; i++) free(dataset->data[i]);
    free(dataset->data); for (int i = 0; i < dataset->cols; i++) free(dataset->featureNames[i]);
    free(dataset->featureNames);
    free(dataset->numValues);
    free(dataset);
}