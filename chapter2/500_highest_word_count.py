from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups


cv = CountVectorizer(stop_words="english", max_features=500)
groups = fetch_20newsgroups()

#print(len(groups.data))
transformed = cv.fit_transform(groups.data)
#print(transformed)
#print(cv.vocabulary_)

print(transformed.toarray().sum(axis=0))

sns.displot(np.log(transformed.toarray().sum(axis=0)))
plt.xlabel('Log Count')
plt.ylabel('Frequency')
plt.title('Distribution Plot of 500 Word Counts')
plt.show()