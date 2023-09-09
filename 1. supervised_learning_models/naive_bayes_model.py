def naive_bayes_model(X_train, Y_train, X_test, Y_test):
    model = GaussianNB()
    model.fit(X_train, Y_train.astype('int'))
    pred = model.predict(X_test)
    print("Naive Bayes MSE:", mean_squared_error(Y_test, pred))