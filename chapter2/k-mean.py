from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
def letters_only(astr):
    return astr.isalpha()


cv = CountVectorizer(stop_words="english", max_features=500)
groups = fetch_20newsgroups()
cleand = []

all_names = set(names.words())
lemmatizer = WordNetLemmatizer()

for post in groups.data:
    cleand.append(' '.join([lemmatizer.lemmatize(word.lower())
                            for word in post.split() if
                            letters_only(word) and word not in all_names]))

transformed = cv.fit_transform(cleand)
km = KMeans(n_clusters=20)
km.fit(transformed)
labels = groups.target
plt.scatter(labels, km.labels_)
plt.xlabel('Newsgroup')
plt.ylabel('Cluster')
plt.show()

unique_labels = np.unique(labels)

# Assign a color to each label
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

# Plot each data point with its corresponding color
for i, label in enumerate(unique_labels):
    cluster_points = km.labels_[labels == label]
    plt.scatter(labels[labels == label], cluster_points, color=colors[i], label=label)

# Add legend

# Set labels and show the plot
plt.xlabel('Newsgroup')
plt.ylabel('Cluster')
plt.show()