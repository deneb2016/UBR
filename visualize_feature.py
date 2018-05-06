import numpy as np
from sklearn.manifold import TSNE
import pickle
from matplotlib import pyplot as plt
val_data = pickle.load(open('extracted_features', 'rb'))
tval_data = pickle.load(open('tval_extracted_features', 'rb'))

X = np.concatenate((val_data['feature'], tval_data['feature']))
print(X.shape)
# X_embedded = TSNE(n_components=2, verbose=3, perplexity=100).fit_transform(X)
# pickle.dump(X_embedded, open('embedded_feature', 'wb'))

X_embedded = pickle.load(open('embedded_feature', 'rb'))
print(X_embedded.shape)
print(len(val_data['feature']))

mean1 = X_embedded[:len(val_data['feature']), :].mean(0)
mean2 = X_embedded[len(val_data['feature']):, :].mean(0)

for i in range(len(X_embedded)):
    if i < len(val_data['feature']):
        plt.plot(X_embedded[i, 0], X_embedded[i, 1], 'o', color='blue')
    else:
        plt.plot(X_embedded[i, 0], X_embedded[i, 1], 'o', color='red')

plt.plot(mean1[0], mean1[1], 'o', color='black')
plt.plot(mean2[0], mean2[1], 'o', color='black')

plt.show()