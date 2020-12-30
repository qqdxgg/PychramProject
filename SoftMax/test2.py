from mxnet import init, nd
from mxnet.gluon import nn
import time


class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        # 实际的初始化逻辑在此省略了


net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10))

net.initialize(init=MyInit())
print('尚未执行前向计算')
time.sleep(4)
print('开始执行前向计算')
X = nd.random.uniform(shape=(2, 20))
Y = net(X)
