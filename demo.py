from keras import optimizers, losses
from example_network import example_network

model = example_network(input_shape=(32,32,3))
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
model.fit(x=train_x, y=train_y, epochs=1, verbose=1)
loss, acc = model.evaluate(x=test_x, y=test_y)
print('Test accuracy is {:.4f}'.format(acc))