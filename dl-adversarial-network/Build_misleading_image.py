import tensorflow as tf
import Layers
import matplotlib.pyplot as plt
import numpy as np
import cv2


LoadModel = True
trainModel = True

experiment_size = 1
fake_label = [1.0, 0.0]

# Load and transform the image
im = cv2.imread("./images/input.png")
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# im = cv2.resize(im, (48, 48))
im = (im - 128.0) / 256.0


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


def train_filter(model, optimizer, X, label, DX, beta):
    with tf.GradientTape() as tape:
        y = model(X + DX)
        y = tf.nn.log_softmax(y)
        diff = label * y
        loss = -tf.reduce_sum(diff) + beta*tf.nn.l2_loss(DX)
        grads = tape.gradient(loss, [DX])
        optimizer.apply_gradients(zip(grads, [DX]))
    return loss


print("== TRAIN DX ==")
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
simple_cnn = ConvNeuralNet()

if LoadModel:
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=simple_cnn)
    ckpt.restore('./saved_model-3-1')

X = tf.constant(im, dtype=np.float32)
DXinit = tf.constant(0.0, shape=[48, 48])
DX = tf.Variable(DXinit)


"""
# On prédit, pour vérifier qu'au départ, le modèle prédit bien une femme
Xmod = X + DX
y_pred = simple_cnn(Xmod, False)
print(y_pred)
print(tf.argmax(y_pred, 1).numpy())
"""

if trainModel:
    for iter in range(1000):
        loss = train_filter(simple_cnn, optimizer, X, fake_label, DX, 0.5)

        if iter % 100 == 0:
            print("iter= %6d - loss= %f" % (iter, loss))

Xmod = X + DX
Xmod_render = Xmod.numpy() * 256 + 128
plt.imshow(Xmod_render, cmap='gray')
plt.show()
cv2.imwrite('./images/misleading_image.png', Xmod_render)

DX_render = DX.numpy() * 256 + 128
plt.imshow(DX_render, cmap='gray')
plt.show()
cv2.imwrite('./images/filter_added.png', DX_render)

y_pred = simple_cnn(Xmod)
print(y_pred)
print(tf.argmax(y_pred, 1).numpy())


print("DONE")

