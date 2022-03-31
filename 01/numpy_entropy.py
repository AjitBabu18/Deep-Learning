#!/usr/bin/env python3
import argparse
import math
import collections
import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_path", default="numpy_entropy_data.txt", type=str, help="Data distribution path.")
parser.add_argument("--model_path", default="numpy_entropy_model.txt", type=str, help="Model distribution path.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace):
    # TODO: Load data distribution, each line containing a datapoint -- a string.
    lis1 = []
    with open(args.data_path, "r") as data:
        for line in data:
            line = line.rstrip("\n")
            lis1.append(line)
            # TODO: Process the line, aggregating data with built-in Python
            # data structures (not NumPy, which is not suitable for incremental
            # addition and string mapping).
    #dat = np.array(lis1)
    #print(dat)
    l, c = np.unique(lis1, return_counts=True)
    c = c/len(lis1)
    
    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. Alternatively,
    # the NumPy array might be created after loading the model distribution.

    # TODO: Load model distribution, each line `string \t probability`.
    mod = np.zeros(c.shape)
    with open(args.model_path, "r") as model:
        for line in model:
            line = line.rstrip("\n")
            lis2 = line.split("\t")
            #print(lis2)
            # TODO: process the line, aggregating using Python data structures
            if np.where(l==lis2[0]):
                mod[np.where(l==lis2[0])] = float(lis2[1])
    # TODO: Create a NumPy array containing the model distribution.
    
    

    # TODO: Compute the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    #def entropy_H():
    
    entropy = 0
    for i in c:
         entropy = entropy - (i * np.log(i))
    #entropy = [-i*math.log(i,2) for i in [j/len(lis2) for j in collections.Counter(lis2).items()]]
    #entropy = - (float(data.count(x)) / len(data)) * math.log(float(data.count(x)) / len(data), 2)
    # TODO: Compute cross-entropy H(data distribution, model distribution).
    # When some data distribution elements are missing in the model distribution,
    # return `np.inf`.
    crossentropy = 0
    for i in range(np.prod(c.shape)):
        crossentropy = crossentropy - (c[i] * np.log(mod[i]))
    # TODO: Compute KL-divergence D_KL(data distribution, model_distribution),
    # again using `np.inf` when needed.
    kl_divergence = 0
    for i in range(np.prod(c.shape)):
        kl_divergence = kl_divergence + (c[i] * (np.log(c[i]) - np.log(mod[i])))
    # Return the computed values for ReCodEx to validate
    return entropy, crossentropy, kl_divergence

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, crossentropy, kl_divergence = main(args)
    print("Entropy: {:.2f} nats".format(entropy))
    print("Crossentropy: {:.2f} nats".format(crossentropy))
    print("KL divergence: {:.2f} nats".format(kl_divergence))
    
