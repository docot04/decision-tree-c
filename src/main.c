// #include <stdio.h>
// #include "data/dataset.h"
// #include "training/decision_tree.h"

// int main() {
//     Dataset *dataset = loadCSV("dataset.csv");
//     if (!dataset) return 1;
//     printf("Rows: %d\n", dataset->rows);
//     printf("Columns: %d\n\n", dataset->cols);
//     printDataset(dataset);
//     // int labels[] = {0, 0, 0, 1, 1, 1};
//     int labels[] = {0, 0, 0, 0, 1, 1};
//     printf("\nEntropy: %f\n", entropy(labels, 6));
//     freeDataset(dataset);
//     return 0;
// }

#include <stdio.h>
#include "data/dataset.h"
#include "model/decision_tree.h"

int main() {
    Dataset *dataset = loadCSV("dataset.csv");
    if (!dataset) return 1;
    int targetIndex = dataset->cols - 1;
    printf("Rows: %d\n", dataset->rows);
    printf("Columns: %d\n\n", dataset->cols);
    printDataset(dataset);
    printf("\nInformation Gain:\n");
    for (int feature = 0; feature < targetIndex; feature++) {
        double gain = informationGain(dataset, feature, targetIndex);
        printf("%s: %.6f\n", dataset->featureNames[feature], gain);
    }
    freeDataset(dataset);
    return 0;
}