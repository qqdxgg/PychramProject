import d2lzh as d2l
from mxnet import autograd, nd
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
for X, y in train_iter:
    print(X.shape, y.shape)
    break
num_inputs = 784
num_outputs = 10
w = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))
b = nd.zeros(num_outputs)
w.attach_grad()
b.attach_grad()


def softmax(X):
    xexp = X.exp()
    partition = xexp.sum(axis=1, keepdims=True)
    return xexp / partition


def net(X):
    return softmax(nd.dot(X.reshape(-1, num_inputs), w) + b)


def cross_entropy(y_hat, y):
    return -nd.pick(y_hat, y).log()


def accuracy_rate(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean.asscalar()


def evaluate_accuracy(diter, net):
    acc_num, total = 0.0, 0
    for X, y in diter:
        y1 = y.astype('float32')
        acc_num += (net(X).argmax(axis=1) == y1).sum().asscalar()
        total += y.size
    return acc_num / total
def show_fashion_mnist(images, labels):
    """Plot Fashion-MNIST images with labels."""
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)


num_epochs, lrate = 3, 0.1


def sfm_train(net, train_iter, test_iter, loss, num_epochs, batch_size,
              lr=None, param=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            if trainer is None:
                d2l.sgd(param, lr, batch_size)
            else:
                trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d,loss %.4f,train_acc %.3f,test_acc %.3f' %
              (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


sfm_train(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size,
          lrate, [w, b])

# for X, y in test_iter:
#     true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
#     # x1 = net(X).argmax(axis=1).asnumpy()
#     # print(x1.dtype,y.asnumpy().dtype)
#     pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).astype('int32').asnumpy())
#     titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
#     print(titles[0:9])
#     d2l.show_fashion_mnist(X[0:9], titles[0:9])
#     break

for X, y in test_iter:
    break



true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
print(titles)
d2l.show_fashion_mnist(X[0:9], titles[0:9])




