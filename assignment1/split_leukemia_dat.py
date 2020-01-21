import numpy as np
import pandas as pd

examples = pd.read_csv("leukemia.dat")
# shuffle input
examples = examples.sample(frac=1) 

#splitting ALL examples into AML="yes" examples and AML="no" examples, so that the 80-20 splitting mantain the same distribution of output class
aml_mask = (examples["AML"] == "yes")
aml_examples = examples[aml_mask]
perc_aml = round(len(aml_examples)/len(examples), 2)

no_aml_mask = (examples["AML"] == "no")
no_aml_examples = examples[no_aml_mask]
perc_no_aml = round(len(no_aml_examples)/len(examples), 2)

print("Total number of examples {0}".format(len(examples)))

print("\n\nTotal number of AML=\"yes\" examples {0} ({1})".format(len(aml_examples), perc_aml))
print("Total number of AML=\"no\" examples {0} ({1})".format(len(no_aml_examples), perc_no_aml))


#fractions of training and test set
fractions = np.array([0.8, 0.2])

# split aml_examples into 2 parts (0.8 train, 0.2 test)
aml_train, aml_test = np.array_split(
    aml_examples, (fractions[:-1].cumsum() * len(aml_examples)).astype(int))
perc_aml_train = round(len(aml_train)/len(aml_examples), 2)
perc_aml_test = round(len(aml_test)/len(aml_examples), 2)

print("\n\nTotal number of AML=\"yes\" examples for training {0} ({1})".format(len(aml_train), perc_aml_train))
print("Total number of AML=\"yes\" examples for test {0} ({1})".format(len(aml_test), perc_aml_test))

# split no_aml_examples into 2 parts (0.8 train, 0.2 test)
no_aml_train, no_aml_test = np.array_split(
    no_aml_examples, (fractions[:-1].cumsum() * len(no_aml_examples)).astype(int))
no_perc_aml_train = round(len(no_aml_train)/len(no_aml_examples), 2)
no_perc_aml_test = round(len(no_aml_test)/len(no_aml_examples), 2)

print("\n\nTotal number of AML=\"no\" examples for training {0} ({1})".format(len(no_aml_train), no_perc_aml_train))
print("Total number of AML=\"no\" examples for test {0} ({1})".format(len(no_aml_test), no_perc_aml_test))

print("\n")
#concat the aml_examples for training with the no_aml_examples for training
examples_train = aml_train.append(no_aml_train)
perc_examples_train = round(len(examples_train)/len(examples), 2)
#shuffle examples for training
examples_train = examples_train.sample(frac=1) 

print("Total number of examples for training {0} ({1})".format(len(examples_train), perc_examples_train))

#concat the aml_examples for testing with the no_aml_examples for testing
examples_test = aml_test.append(no_aml_test)
perc_examples_test = round(len(examples_test)/len(examples), 2)
#shuffle examples for testing
examples_test = examples_test.sample(frac=1) 

print("Total number of examples for testing {0} ({1})".format(len(examples_test), perc_examples_test))

#write down the two datasets into file
examples_train.to_csv("leukemia_train.dat", index=False)
examples_test.to_csv("leukemia_test.dat", index=False)