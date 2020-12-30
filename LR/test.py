from mxnet import autograd, nd, init, gluon
from mxnet.gluon import data as gdata, loss as gloss, nn

imput_dim = 2
example_num = 1000
ture_w = [2, -4.3]
ture_b = 3
features = nd.random.normal(scale=1, shape=(example_num, imput_dim))
labels = nd.dot(features, nd.array(ture_w).reshape(1, 2).T) + ture_b
labels += nd.random.normal(scale=0.1, shape=labels.shape)
batch_size = 10
dataset = gdata.ArrayDataset(features,labels)
data_itert = gdata.DataLoader(dataset,batch_size,shuffle=True)

# 定义模型
lr_net = nn.Sequential()
lr_net.add(nn.Dense(1))
lr_net.initialize(init.Normal(sigma=0.01))
loss = gloss.L2Loss()
trianer = gluon.Trainer(lr_net.collect_params(),'sgd',{'learning_rate':0.03})
num_epochs = 5
for epoch in range(1,num_epochs+1):
    for x,y in data_itert:
        with autograd.record():
            l = loss(lr_net(x),y)
        l.backward()
        trianer.step(batch_size)
    l = loss(lr_net(features),labels)
    print(type(l))
    print('epoch:%d , loss:%f'%(epoch,l.mean().asnumpy()))

dense = lr_net[0]
print(ture_w,dense.weight.data())
print(ture_b,dense.bias.data())