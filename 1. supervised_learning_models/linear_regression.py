def linear_regression_model(X_train, Y_train, X_test, Y_test):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    print("Linear Regression MSE:", mean_squared_error(Y_test, pred))