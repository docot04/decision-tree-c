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