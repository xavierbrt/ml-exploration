import tensorflow as tf
import Layers
import matplotlib.pyplot as plt
import numpy as np
import cv2



# Load and transform image
im = cv2.imread("./images/input.png")
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# im = cv2.resize(im, (48, 48))
im = (im - 128.0) / 256.0
# Load and transform image
im2 = cv2.imread("./images/misleading_image.png")
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
# im2 = cv2.resize(im2, (48, 48))
im2 = (im2 - 128.0) / 256.0





class ConvNeuralNet(tf.Module):
    def __init__(self):
        self.unflat = Layers.unflat(48, 48, 1)
        self.cv1 = Layers.conv(output_dim=3, filterSize=5, stride=1)
        self.mp = Layers.maxpool(2)
        self.cv2 = Layers.conv(output_dim=6, filterSize=5, stride=1)
        self.cv3 = Layers.conv(output_dim=12, filterSize=5, stride=1)
        self.flat = Layers.flat()
        self.fc = Layers.fc(2)

    def __call__(self, x):
        x = self.unflat(x)
        x = self.cv1(x)
        x = self.mp(x)
        x = self.cv2(x)
        x = self.mp(x)
        x = self.cv3(x)
        x = self.mp(x)
        x = self.flat(x)
        x = self.fc(x)
        return x


print("== LOAD MODEL ==")
optimizer = tf.optimizers.Adam(1e-3)
simple_cnn = ConvNeuralNet()

# Load model:
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=simple_cnn)
ckpt.restore('./saved_model-3-1')

print("== PREDICT ==")
# Prediction sur nos 2 images
data = np.array([im, im2], dtype=np.float32)
y_pred = simple_cnn(data)
result = tf.argmax(y_pred, 1).numpy()
print(y_pred)
print(result)

# Plot de nos 4 images
for i in range(0, 2):
    plt.subplot(1, 2, i + 1)
    img = np.reshape(data[i], (48, 48))
    plt.title(result[i])
    plt.imshow(img, cmap='gray')
plt.show()

"""
[[ 1.7100327  2.0411801]
 [ 6.490074  -3.2798302]]
"""

## BROUILLON

# train = ds.DataSet('../DataBases/data_%dk.bin' % experiment_size, '../DataBases/gender_%dk.bin' % experiment_size, 1000 * experiment_size)
# test = ds.DataSet('../DataBases/data_test10k.bin', '../DataBases/gender_test10k.bin', 10000)

"""
# Prediction sur nos 4 images
data = np.array([im, im2, im3, im4], dtype=np.float32)
y_pred = simple_cnn(data, False)
print(y_pred)
print(tf.argmax(y_pred, 1))
y_pred2 = tf.argmax(y_pred, 1).numpy()

# Plot de nos 4 images
for i in range(0,4):
    plt.subplot(1,4,i+1)
    img = np.reshape(data[i],(48,48))
    plt.title(y_pred2[i])
    plt.imshow(img, cmap='gray')
plt.show()"""

"""im_flat = im.flatten()
im_flat2 = im2.flatten()
im_flat3 = im3.flatten()
images = np.array([im_flat, im_flat2, im_flat3], dtype=np.float32)
"""

"""
test.batchSize = 1000
print(test.mean_accuracy(simple_cnn) * 100)
curBatchSize = min(test.batchSize, test.nbdata - 0)
res = simple_cnn(test.data[0:0+curBatchSize,:], False)
print(res)
print(tf.argmax(res, 1))
#print(test.label[0:curBatchSize])
"""

"""
# Plot de 15 images de train
for i in range(1,16):
    plt.subplot(4,4,i)
    img = np.reshape(train.data[i],(48,48))
    plt.title(train.label[i])
    plt.imshow(img, cmap='gray')
plt.subplot(4,4,16)
plt.imshow(im3, cmap='gray')
plt.show()
"""





""" IMPLEMENTATION DIFFERENTE DU RESEAU DE NEURONES (test)
Doc: https://missinglink.ai/guides/tensorflow/tensorflow-conv2d-layers-practical-guide/


#x = conv2d(x, biases['bc1'], 1)
        

def conv2d(x, b, strides):
    w_init = tf.random.truncated_normal([5, 5, x.shape[3], 3], stddev=0.1, seed=8)
    w = tf.Variable(w_init)
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME') + b
    return tf.nn.relu(x)


#filterSize = 5
#filterSize, filterSize, x.shape[3], self.output_dim]
#1 3 3 6 6 12

weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random.truncated_normal([5, 5, 1, 3], stddev=0.1, seed=8)),
}
biases = {
    'bc1': tf.Variable(tf.constant(0.0, shape=[3])),
}
"""