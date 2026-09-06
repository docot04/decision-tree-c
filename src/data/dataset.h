#ifndef DATASET_H
#define DATASET_H

typedef struct {
    int rows;               // no. of datapoints
    int cols;               // no. of features
    int **data;             // dataset pointer
    char **featureNames;    // feature names (for display)
    int *numValues;         // unique values per feature
    char ***valueNames;     // to retain information about categorical mappings
} Dataset;

// categorically encodes each new value pre feature
static int getValueIndex(char **values, int *count, const char *value);

// reads a CSV file and loads its encoded value indices into memory 
Dataset *loadCSV(const char *filename);

// DEBUGGING: print the dataset with value indices
void printDataset(Dataset *dataset);

// free the entire dataset from memory
void freeDataset(Dataset *dataset);

#endif