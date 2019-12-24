import numpy as np
import itertools

class KFoldSplit(object):
    def __init__(self, folds):
        self._index = 0
        self._folds = folds

    def __iter__(self):
        return self

    def __next__(self):
        k = len(self._folds)
        if self._index < k:
            train_idx = []
            test_idx = []
            for i in range(k):
                if i == self._index:
                    test_idx = self._folds[i]
                else:
                    train_idx.append(self._folds[i])
            self._index = self._index + 1
            train_idx_flat = list(itertools.chain(*train_idx))
            return train_idx_flat, test_idx
        raise StopIteration


class CrossValidator(object):
    def __init__(self, n_splits=3, random_state=None, shuffle=None):
        self._n_splits = n_splits
        self._rnd = np.random.RandomState(random_state)
        self._shuffle = shuffle

    def split(self, X):
        n = X.shape[0]
        indices = np.arange(0, n)

        if self._shuffle:
            self._rnd.shuffle(indices)

        fold_size = int(np.round(n/self._n_splits))
        folds = []
        picked = 0
        for i in range(self._n_splits):
            if i == (self._n_splits - 1):
                fold = list(indices[picked:])
            else:
                fold = list(indices[picked:picked+fold_size])
            folds.append(fold)
            picked = picked + len(fold)
        if len(folds) != self._n_splits:
            raise OverflowError

        return KFoldSplit(folds)

if __name__=="__main__":
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    val = CrossValidator(n_splits=3, shuffle=True)
    for train_idx, test_idx in val.split(x):
        print("Train")
        print(train_idx)
        print("Selected")
        print(np.take(x, train_idx))
        print("Test")
        print(test_idx)
        print("Selected")
        print(np.take(x, test_idx))
        print("")
