import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

#create a neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'), #change the number of hidden layers & hidden nodes
    tf.keras.layers.Dense(128, activation='sigmoid'), #change the activation function " relu, sigmoid, softmax "
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(10, activation='softmax'),
    
])

#compile the model
model.compile(optimizer='adam',
              loss= 'mean_squared_error',  # change the loss function
              metrics=['accuracy'])

#train the model and evaluate
model.fit(x_train, y_train, epochs=5) #increase the epoch
model.evaluate(x_test, y_test)
