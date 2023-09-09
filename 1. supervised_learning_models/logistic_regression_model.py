def logistic_regression_model(X_train, Y_train, X_test, Y_test):
    model = LogisticRegression()
    model.fit(X_train, Y_train.astype('int'))
    pred = model.predict(X_test)
    print("Logistic Regression MSE:", mean_squared_error(Y_test, pred))