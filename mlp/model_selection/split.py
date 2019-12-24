import numpy as np

def train_test_split(X, y, test_size=0.33, random_state=None):
    n = X.shape[0]
    indices = np.arange(0, n)

    rnd = np.random.RandomState(random_state)
    rnd.shuffle(indices)

    split_point = int(np.round(n*(1-test_size)))
    X_train, y_train = np.take(X, indices[0:split_point], axis=0), np.take(y, indices[0:split_point])
    X_test, y_test = np.take(X, indices[split_point:], axis=0), np.take(y, indices[split_point:])
    return X_train, y_train, X_test, y_test


def next_batch(X, y, batch_size=32, random_state = None):
    n = X.shape[0]
    indices = np.arange(0, n)

    rnd = np.random.RandomState(random_state)
    rnd.shuffle(indices)

    if batch_size is None:
        batch_size = n
    num_batches = int(np.floor(n/batch_size))
    for i in range(0, batch_size*num_batches, batch_size):
        yield np.take(X, indices[i:(i+batch_size)], axis=0), np.take(y, indices[i:(i+batch_size)], axis=0)

if __name__=="__main__":
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array([-1, -2, -3, -4, -5, -6, -7, -8])
    for X_batch, y_batch in next_batch(x, y, 3):
        print("X")
        print(X_batch)
        print("y")
        print(y_batch)

