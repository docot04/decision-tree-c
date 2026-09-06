#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"

// HELPER: string -> [len] [string]
static void writeString(FILE *file, const char *string) {
    int length = strlen(string);
    fprintf(file, "%d %s", length, string);
}

// HELPER: [len] [string] -> string 
static char *readString(FILE *file) {
    int length;
    if (fscanf(file, "%d", &length) != 1) return NULL;
    int c = fgetc(file);
    if (c != ' ') return NULL;
    char *string = malloc((length + 1) * sizeof(char));
    if (!string) return NULL;
    if (fread(string, sizeof(char), length, file) != (size_t)length) {
        free(string);
        return NULL;
    }
    string[length] = '\0';
    return string;
}

// HELPER: saves nodes
static void saveNode(FILE *file, TreeNode *node) {
    if (node->isLeaf) {
        fprintf(file, "LEAF %d\n", node->classLabel);
        return;
    }
    fprintf(file, "NODE FEATURE %d\n", node->featureIndex);
    for (int i = 0; i < node->numChildren; i++) {
        fprintf(file, "CHILD %d\n", node->featureValues[i]);
        saveNode(file, node->children[i]);
    }
    fprintf(file, "END\n");
}

// HELPER: convert dataset into the ".dt" format
static void saveDataset(FILE *file, Dataset *dataset) {
    int featureCount = dataset->cols - 1;
    fprintf(file, "FEATURES %d\n\n", featureCount);
    for (int col = 0; col < featureCount; col++) {
        fprintf(file, "FEATURE %d ", col);
        writeString(file, dataset->featureNames[col]);
        fprintf(file, " %d\n", dataset->numValues[col]);
        for (int value = 0; value < dataset->numValues[col]; value++) {
            fprintf(file, "VALUE %d ", value);
            writeString(file, dataset->valueNames[col][value]);
            fprintf(file, "\n");
        }
        fprintf(file, "\n");
    }
    int targetIndex = dataset->cols - 1;
    fprintf(file, "TARGET ");
    writeString(file, dataset->featureNames[targetIndex]);
    fprintf(file, " %d\n", dataset->numValues[targetIndex]);
    for (int value = 0; value < dataset->numValues[targetIndex]; value++) {
        fprintf(file, "CLASS %d ", value);
        writeString(file, dataset->valueNames[targetIndex][value]);
        fprintf(file, "\n");
    }
    fprintf(file, "\n");
}

int saveModel(const char *filename, TreeNode *root, Dataset *dataset) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("Could not create model file\n");
        return 0;
    }
    fprintf(file,"DTREE 1\n\n");
    saveDataset(file, dataset);
    fprintf(file, "TREE\n");
    saveNode(file, root);
    fclose(file);
    return 1;
}

// HELPER: loads from ".dt" file
static Dataset *loadDataset(FILE *file) {

    // create Dataset structure
    char token[32];
    int featureCount;
    if (fscanf(file, "%31s %d", token, &featureCount) != 2) return NULL;
    if (strcmp(token, "FEATURES") != 0) return NULL;
    Dataset *dataset = malloc(sizeof(Dataset));
    if (!dataset) return NULL;
    dataset->cols = featureCount + 1;
    dataset->rows = 0;
    dataset->data = NULL;
    dataset->featureNames = malloc(dataset->cols * sizeof(char *));
    dataset->numValues = malloc(dataset->cols * sizeof(int));
    dataset->valueNames = malloc(dataset->cols * sizeof(char **));
    if (!dataset->featureNames || !dataset->numValues || !dataset->valueNames) {
        free(dataset->featureNames);
        free(dataset->numValues);
        free(dataset->valueNames);
        free(dataset);
        return NULL;
    }

    // load features
    for (int i = 0; i < featureCount; i++) {
        int featureIndex;
        int numValues;
        if (fscanf(file, "%31s %d", token, &featureIndex) != 2) return NULL;
        if (strcmp(token, "FEATURE") != 0) return NULL;
        dataset->featureNames[featureIndex] = readString(file);
        if (!dataset->featureNames[featureIndex]) return NULL;
        if (fscanf(file, "%d", &numValues) != 1) return NULL;
        dataset->numValues[featureIndex] = numValues;
        dataset->valueNames[featureIndex] = malloc(numValues * sizeof(char *));
        if (!dataset->valueNames[featureIndex]) return NULL;

        // load values
        for (int j = 0; j < numValues; j++) {
            int valueIndex;
            if (fscanf(file, "%31s %d", token, &valueIndex) != 2) return NULL;
            if (strcmp(token, "VALUE") != 0) return NULL;
            dataset->valueNames[featureIndex][valueIndex] = readString(file);
            if (!dataset->valueNames[featureIndex][valueIndex]) return NULL;
        }
    }
    
    // load target
    int targetIndex;
    int numClasses;
    if (fscanf(file, "%31s", token) != 1) return NULL;
    if (strcmp(token, "TARGET") != 0) return NULL;
    targetIndex = dataset->cols - 1;
    dataset->featureNames[targetIndex] = readString(file);
    if (!dataset->featureNames[targetIndex]) return NULL;
    if (fscanf(file, "%d", &numClasses) != 1) return NULL;
    dataset->numValues[targetIndex] = numClasses;
    dataset->valueNames[targetIndex] = malloc(numClasses * sizeof(char *));
    if (!dataset->valueNames[targetIndex]) return NULL;
    for (int i = 0; i < numClasses; i++) {
        int classIndex;
        if (fscanf(file, "%31s %d", token, &classIndex) != 2) return NULL;
        if (strcmp(token, "CLASS") != 0) return NULL;
        dataset->valueNames[targetIndex][classIndex] = readString(file);
        if (!dataset->valueNames[targetIndex][classIndex]) return NULL;
    }
    return dataset;
}

// HELPER: loads nodes
static TreeNode *loadNode(FILE *file, Dataset *dataset) {
    char token[32];

    // read node type
    if (fscanf(file, "%31s", token) != 1) return NULL;

    // leaf node (LEAF 0)
    if (strcmp(token, "LEAF") == 0) {
        int classLabel;
        if (fscanf(file, "%d", &classLabel) != 1) return NULL;
        TreeNode *node = malloc(sizeof(TreeNode));
        if (!node) return NULL;
        node->isLeaf = 1;
        node->featureIndex = -1;
        node->classLabel = classLabel;
        node->numChildren = 0;
        node->featureValues = NULL;
        node->children = NULL;
        return node;
    }

    // decision node (NODE FEATURE 0)
    if (strcmp(token, "NODE") == 0) {
        if (fscanf(file, "%31s", token) != 1) return NULL;
        if (strcmp(token, "FEATURE") != 0) return NULL;
        int featureIndex;
        if (fscanf(file, "%d", &featureIndex) != 1) return NULL;
        int numChildren = dataset->numValues[featureIndex];
        
        // allocate node
        TreeNode *node = malloc(sizeof(TreeNode));
        if (!node) return NULL;
        node->isLeaf = 0;
        node->featureIndex = featureIndex;
        node->classLabel = -1;
        node->numChildren = numChildren;

        // allocate children
        node->featureValues = malloc(numChildren * sizeof(int));
        node->children = malloc(numChildren * sizeof(TreeNode *));
        if (!node->featureValues || !node->children) {
            free(node->featureValues);
            free(node->children);
            free(node);
            return NULL;
        }
        for (int i = 0; i < numChildren; i++) {
            node->featureValues[i] = -1;
            node->children[i] = NULL;
        }

        // load children
        for (int i = 0; i < numChildren; i++) {
                if (fscanf(file, "%31s", token) != 1) {
                freeTree(node);
                return NULL;
            }
            if (strcmp(token, "CHILD") != 0) {
                freeTree(node);
                return NULL;
            }
            int featureValue;
            if (fscanf(file, "%d", &featureValue) != 1) {
                freeTree(node);
                return NULL;
            }
            node->featureValues[i] = featureValue;
            node->children[i] = loadNode(file, dataset);
            if (!node->children[i]) {
                freeTree(node);
                return NULL;
            }
        }

        // END token
        if (fscanf(file, "%31s", token) != 1) {
            freeTree(node);
            return NULL;
        }
        if (strcmp(token, "END") != 0) {
            freeTree(node);
            return NULL;
        }

        return node;
    }

    // unknown token
    printf("Unknown node token: %s\n", token);
    return NULL;
}


TreeNode *loadModel(const char *filename, Dataset **dataset) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Could not open model file\n");
        return NULL;
    }

    // model header (DTREE 1)
    char token[32];
    int version;
    if (fscanf(file, "%31s %d", token, &version) != 2) {
        printf("Invalid model header\n");
        fclose(file);
        return NULL;
    }
    if (strcmp(token, "DTREE") != 0) {
        printf("Not a DTREE model\n");
        fclose(file);
        return NULL;
    }
    if (version != 1) {
        printf("Unsupported model version: %d\n", version);
        fclose(file);
        return NULL;
    }

    // load dataset (expects file pointer at FEATURES x)
    *dataset = loadDataset(file);
    if (!*dataset) {
        printf("Failed to load dataset\n");
        fclose(file);
        return NULL;
    }

    // read the tree
    if (fscanf(file, "%31s", token) != 1) {
        printf("Missing TREE section\n");
        freeDataset(*dataset);
        *dataset = NULL;
        fclose(file);
        return NULL;
    }
    if (strcmp(token, "TREE") != 0) {
        printf("Expected TREE, found %s\n", token);
        freeDataset(*dataset);
        *dataset = NULL;
        fclose(file);
        return NULL;
    }

    // load tree recursively
    TreeNode *root = loadNode(file, *dataset);
    if (!root) {
        printf("Failed to load tree\n");
        freeDataset(*dataset);
        *dataset = NULL;
        fclose(file);
        return NULL;
    }

    fclose(file);
    printf("Model loaded successfully!\n");
    return root;
}