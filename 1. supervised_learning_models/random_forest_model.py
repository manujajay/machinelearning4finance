def random_forest_model(X_train, Y_train, X_test, Y_test):
    model = RandomForestRegressor()
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    print("Random Forest MSE:", mean_squared_error(Y_test, pred))