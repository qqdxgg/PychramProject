import collections
import d2lzh
import math
from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import sys
import zipfile
import time
import random

# with zipfile.ZipFile('./ptb.zip','r') as zin:
#     zin.extractall('./')

with open('./ptb/ptb.train.txt', 'r') as f:
    lines = f.readlines()
    # print(type(lines),len(lines))
    # print(lines)
    raw_data = [l.split() for l in lines]
    word_single = [tx for line in raw_data for tx in line]
    # print(len(word_single))
    # print(type(raw_data),len(raw_data))
    # print(raw_data)

counter = collections.Counter([t for st in raw_data for t in st])
counter = dict(filter(lambda x: x[1] > 5, counter.items()))

idx2token = [tk for tk, _ in counter.items()]
token2idx = {tk: idx for idx, tk in enumerate(idx2token)}
dataset = [[token2idx[tx] for tx in st if tx in token2idx] for st in raw_data]
numlgr5 = sum([len(st) for st in dataset])
# print(token2idx)
# print(numlgr5)
def discard(idx):
    return random.uniform(0, 1) < 1 - math.sqrt(
        1e-4 / counter[idx2token[idx]] * numlgr5)

subsampled_dataset = [[tk for tk in st if not discard(tk)] for st in dataset]

def get_centers_and_contexts(dataset, max_window_size):
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:  # 每个句子至少要有2个词才可能组成一对“中心词-背景词”
            continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, center_i - window_size),
                                 min(len(st), center_i + 1 + window_size)))
            indices.remove(center_i)  # 将中心词排除在背景词之外
            contexts.append([st[idx] for idx in indices])
    return centers, contexts

tiny_dataset = [list(range(7)), list(range(7, 10))]

# cen,context = get_centers_and_contexts(tiny_dataset,2)
# print(cen,context)
all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)
# print(sum([len(st) for st in all_contexts]))
# print(len([t for st in all_contexts for t in st]))
def get_negatives(all_contexts, sampling_weights, K):
    all_negatives, neg_candidates, i = [], [], 0
    population = list(range(len(sampling_weights)))
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            if i == len(neg_candidates):
                # 根据每个词的权重（sampling_weights）随机生成k个词的索引作为噪声词。
                # 为了高效计算，可以将k设得稍大一点
                i, neg_candidates = 0, random.choices(
                    population, sampling_weights, k=int(1e5))
            neg, i = neg_candidates[i], i + 1
            # 噪声词不能是背景词
            if neg not in set(contexts):
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

sampling_weights = [counter[w]**0.75 for w in idx2token]
all_negatives = get_negatives(all_contexts, sampling_weights, 5)
# print(all_negatives)
def batchify(data):
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (nd.array(centers).reshape((-1, 1)), nd.array(contexts_negatives),
            nd.array(masks), nd.array(labels))

batch_size = 512
num_workers = 0 if sys.platform.startswith('win32') else 4
dataset = gdata.ArrayDataset(all_centers, all_contexts, all_negatives)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True,
                             batchify_fn=batchify, num_workers=num_workers)

def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = nd.batch_dot(v, u.swapaxes(1, 2))
    return pred


loss = gloss.SigmoidBinaryCrossEntropyLoss()


pred = nd.array([[1.5, 0.3, -1, 2], [1.1, -0.6, 2.2, 0.4]])
# 标签变量label中的1和0分别代表背景词和噪声词
label = nd.array([[1, 0, 0, 0], [1, 1, 0, 0]])
mask = nd.array([[1, 1, 1, 1], [1, 1, 1, 0]])  # 掩码变量
embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(idx2token), output_dim=embed_size),
        nn.Embedding(input_dim=len(idx2token), output_dim=embed_size))
def train(net, lr, num_epochs):
    ctx = d2lzh.try_gpu()
    net.initialize(ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    for epoch in range(num_epochs):
        start, l_sum, n = time.time(), 0.0, 0
        for batch in data_iter:
            center, context_negative, mask, label = [
                data.as_in_context(ctx) for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1])
                # 使用掩码变量mask来避免填充项对损失函数计算的影响
                l = (loss(pred.reshape(label.shape), label, mask) *
                     mask.shape[1] / mask.sum(axis=1))
            l.backward()
            trainer.step(batch_size)
            l_sum += l.sum().asscalar()
            n += l.size
        print('epoch %d, loss %.2f, time %.2fs'
              % (epoch + 1, l_sum / n, time.time() - start))

train(net, 0.005, 5)

def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data()
    x = W[token2idx[query_token]]
    # 添加的1e-9是为了数值稳定性
    cos = nd.dot(W, x) / (nd.sum(W * W, axis=1) * nd.sum(x * x) + 1e-9).sqrt()
    topk = nd.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # 除去输入词
        print('cosine sim=%.3f: %s' % (cos[i].asscalar(), (idx2token[i])))

get_similar_tokens('chairman', 3, net[0])

