class GaussianNaiveBayes(object):
    """description of class"""
    
    def fit(self, X, y):
        pass

    def predict(self, X):
        # Check if coef and intercept are defined, meaning that model is trained
        try:
            getattr(self, "_theta")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        pass


