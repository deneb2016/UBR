import numpy as np
from sklearn.manifold import TSNE
import pickle
from matplotlib import pyplot as plt
val_data = pickle.load(open('/home/seungkwan/cal_val_pooled_features', 'rb'))
tval_data = pickle.load(open('/home/seungkwan/cal_tval_pooled_features', 'rb'))

X = np.concatenate((val_data['feature'], tval_data['feature']))
print(X.shape)
# X_embedded = TSNE(n_components=2, verbose=3, perplexity=100).fit_transform(X)
# pickle.dump(X_embedded, open('/home/seungkwan/cal_embedded_pooled_feature', 'wb'))

X_embedded = pickle.load(open('/home/seungkwan/cal_embedded_pooled_feature', 'rb'))
print(X_embedded.shape)
print(len(val_data['feature']))

#
# for i in range(len(X_embedded)):
#     if i < len(val_data['feature']):
#         plt.plot(X_embedded[i, 0], X_embedded[i, 1], 'o', color='blue')
#     else:
#         plt.plot(X_embedded[i, 0], X_embedded[i, 1], 'o', color='red')
# mean1 = X_embedded[:len(val_data['feature']), :].mean(0)
# mean2 = X_embedded[len(val_data['feature']):, :].mean(0)
# plt.plot(mean1[0], mean1[1], 'o', color='black')
# plt.plot(mean2[0], mean2[1], 'o', color='black')

X_embedded = X_embedded[:len(val_data['feature']), :]
for cls in range(60):
    c = np.random.uniform(0, 1, 3)

    mask = np.equal(val_data['label'], cls)
    mask = np.expand_dims(mask, 1)

    mask = np.concatenate([mask, mask], 1)

    here = X_embedded[mask].reshape(-1, 2)
    mean = here.mean(0)
    plt.plot(mean[0], mean[1], 'o', color=c)
    print(mean)
    # for i in range(len(val_data['feature'])):
    #     if val_data['label'][i] == cls:
    #         plt.plot(X_embedded[i, 0], X_embedded[i, 1], 'o', color=c)

plt.show()