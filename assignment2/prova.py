import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

l1 = np.array(["Ciao bello come stai", "bello come giornata", "Napoleone ha conquistato una giornata"])

cv = CountVectorizer()
hv = HashingVectorizer(n_features=2**4)
l1_t = (cv.fit_transform(l1)).toarray()
l1_h = (hv.fit_transform(l1)).toarray()

print(l1_t)
print(l1_t.shape)
print(cv.vocabulary_)

print(l1_h)
print(l1_h.shape)


l2 = np.array(["Ciao bello stai", "bello bello bello chiaro", "Napoleone ha conquistato una giornata"])
l2_t = (cv.transform(l2)).toarray()
l2_h = (hv.fit_transform(l2)).toarray()
print(l2_t)
print(l2_t.shape)

print(l2_h)
print(l2_h.shape)