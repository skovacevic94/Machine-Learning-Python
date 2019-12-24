class Pipeline(object):
    def __init__(self, transformers, model):
        self.transformers_ = transformers
        if transformers is None:
            self.transformers_ = []
        self.model_ = model

    def fit(self, X, y):
        for t in self.transformers_:
            X = t.fit_transform(X)
        self.model_.fit(X, y)

    def predict(self, X):
        for t in self.transformers_:
            X = t.transform(X)
        return self.model_.predict(X)
