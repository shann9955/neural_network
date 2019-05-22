from sklearn.neural_network import MLPClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = MLPClassifier(solver='lbfgs', alpha=10.0,
                     hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(X, y)                         
MLPClassifier(activation='relu', alpha=1000.0, batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant', learning_rate_init=0.001, max_iter=500, momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True, solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False, warm_start=True)

predict = clf.predict([[2., 2.], [-1., -2.]])

print (predict)

predict = clf.predict_proba([[2., 2.], [1., 2.]]) 

print (predict)

#predict = clf.predict([[1., 1]])

#print (predict)

#clf.predict_proba([[2., 2.], [1., 2.]])  