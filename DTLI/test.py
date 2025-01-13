import json
import os
import pprint
import sys
import time
import pandas as pd
from tqdm import trange
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from sklearn.linear_model import LinearRegression
import datetime


# 创建 SparkSession
spark = SparkSession.builder \
        .appName("Hudi Example") \
        .config("spark.jars.packages", "org.apache.hudi:hudi-spark3-bundle_2.12:0.9.0") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "2047m") \
        .getOrCreate()

# 读取 Hudi 表
hudi_table_path = "hdfs://localhost:9000/chao"
df = spark.read.format("hudi").load(hudi_table_path)

# 将 Spark DataFrame 转换为 Pandas DataFrame
patient = df.toPandas()
print(patient.columns)
print(patient.head())
# 数据预处理
patient['timestamp'] = pd.to_datetime(patient['timestamp'])
patient = patient.sort_values(by='timestamp')
patient = patient.reset_index(drop=True)

# 数据分析
print(patient.head())
print(patient.describe())
print(patient.shape)

# 可视化数据分布
patientunitstayid = sorted(patient.patientunitstayid.tolist())
plt.scatter(patientunitstayid, [x for x in range(1, len(patientunitstayid) + 1)], s=3)
plt.plot(patientunitstayid, [x for x in range(1, len(patientunitstayid) + 1)], c='red')
plt.show()

def gen_cdf(keys):
    cdf = []
    total = len(keys)
    for i in range(total):
        if i <= total - 2:
            if keys[i] != keys[i + 1]:
                cdf.append(i / total)
            else:
                continue
        else:
            cdf.append(1.0)
    return cdf

patientunitstayid_cdf = gen_cdf(patientunitstayid)
patientunitstayid_uniform = [x for x in range(patientunitstayid[0], patientunitstayid[-1] + 1)]
patientunitstayid_uniform_cdf = gen_cdf(patientunitstayid_uniform)
plt.scatter(patientunitstayid, patientunitstayid_cdf, s=3)
plt.plot(patientunitstayid, patientunitstayid_cdf, c='red')
plt.plot(patientunitstayid_uniform, patientunitstayid_uniform_cdf, c='green')
plt.show()

print(200859 * patientunitstayid_cdf[256])

low = 0
upper = 30
plt.plot(patientunitstayid[low:upper], patientunitstayid_cdf[low:upper], c='gray')
plt.scatter(patientunitstayid[low:upper], patientunitstayid_cdf[low:upper], s=5)

# 2 分布转换

def load(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def save(model, optimizer, path):
    d = {}
    d["model"] = model.state_dict()
    d["optimizer"] = optimizer.state_dict()
    torch.save(d, path)

def save_weights(model, mean, var, args, weight_path):
    with open(weight_path, 'w') as f:
        model.eval()
        f.write('%d\t%d\t%d\n' % (args['input_dim'], args['input_dim'] * args['hidden_dim'], args['num_layers']))
        f.write('%.16f\t%.16f\n' % (mean.item(), var.item()))
        model.save_weights(f)

def train(train_keys, model, optimizer, scheduler, best_loss, args, model_path):
    model.train()
    model = model.to(args['device'])
    last_loss = None
    batch_size = args['batch_dim']
    for epoch in range(args['steps']):
        print('Epoch:{}'.format(epoch))
        start_time = time.time()
        epoch_loss = 0
        num_batches = int(np.ceil(train_keys.shape[0] / batch_size))
        iterator = trange(num_batches, smoothing=0, dynamic_ncols=True)
        for i in iterator:
            l = i * batch_size
            r = np.min([(i + 1) * batch_size, train_keys.shape[0]])
            x = train_keys[l:r].to(args['device'])
            loss = model.loss(x)
            if np.isnan(loss.item()):
                print('Error: Loss is NaN')
                exit()
            if np.isinf(loss.item()):
                print('Error: Loss is INF')
                exit()

            epoch_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args['clip_norm'])
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(loss)

            iterator.set_postfix(
                loss="{:.2f}".format(loss.data.cpu().numpy()), refresh=False
            )

        epoch_time = time.time() - start_time
        print('Time Cost of Epoch {}'.format(epoch_time))

        print('Epoch mean loss:{}'.format(epoch_loss / train_keys.shape[0]))
        if last_loss is not None and (epoch_loss > last_loss
                                      or np.fabs((last_loss - epoch_loss) / last_loss) < 0.01):
            print('Process: Early stop at epoch-{}'.format(epoch))
            break
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save(model, optimizer, model_path)
        last_loss = epoch_loss

    return best_loss

def test(source_keys, model, args):
    model = model.to(args['device'])
    model.eval()
    tran_keys = None
    with torch.no_grad():
        batch_size = 8192
        num_batches = int(np.ceil(source_keys.shape[0] / batch_size))
        iterator = trange(num_batches, smoothing=0, dynamic_ncols=True)
        for i in iterator:
            l = i * batch_size
            r = np.min([(i + 1) * batch_size, source_keys.shape[0]])
            x = source_keys[l:r].to(args['device'])
            z = model(x).to('cpu')
            if torch.isnan(z).any():
                print('Error: Nan keys')
                exit()
            tran_keys = torch.cat((tran_keys, z), 0) if tran_keys is not None else z
    return tran_keys

from assess_quality import evaluate_keys
from Diatrbution_Transformation import DistTransformer
from data_util import sample_data
from numerical_flow import plot_dist

args = {'device': 'cuda:0', 'data_dir': 'data', 'data_name': 'patientunitstayid',
        'seed': 1000000013, 'plot': True, 'log_file': True,
        'encoder_type': 'partition', 'decoder_type': 'sum', 'shifts': 1000000,
        'keep_res': True, 'reduce_dim': -1, 'num_flows': 1,
        'num_layers': 2, 'input_dim': 2, 'hidden_dim': 1,
        'train_ratio': 0.1, 'num_train': 3, 'learning_rate': 1e-1,
        'clip_norm': .1, 'steps': 15, 'patience': 2000,
        'decay': 0.5, 'batch_dim': 4096, 'loss_func': 'normal',
        'load': None, }

if not os.path.exists('checkpoint'):
    os.mkdir('checkpoint')

now = datetime.datetime.now()
time_str = '-%4d-%02d-%02d-%02d-%02d-%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
checkpoint_dir = os.path.join('checkpoint', args['data_name'] + time_str)
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
model_path = os.path.join(checkpoint_dir, 'checkpoint.pt')

# Set random seed
random_seed = args['seed']
seed = random_seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

if args['log_file']:
    log_filename = os.path.join(checkpoint_dir, 'outputs.log')
    log_f = open(log_filename, 'w')
else:
    log_f = None

# Loading data, analyzing data, segmenting data
print('Process: Loading data...')
start_time = time.time()

# 加载数据
load_keys = torch.Tensor(patientunitstayid).reshape(-1, 1)
load_durations = time.time() - start_time
print('Time Cost of Loading data {}'.format(load_durations))

print('Process: Evaluating original keys [{}]...'.format(load_keys.size()))
# evaluate_keys(load_keys.detach().cpu().numpy().squeeze(), log_f=log_f)

print('Process: Using Min-Max Normalization')
# 维度扩张的前序步骤，对应伪代码1-4行
global_mean = torch.min(load_keys)
global_var = (torch.max(load_keys) - torch.min(load_keys)) / args['shifts']
load_keys = (load_keys - global_mean) / global_var
print('Min {} Max {}'.format(torch.min(load_keys), torch.max(load_keys)))
if args['log_file']:
    print('Min {} Max {}'.format(torch.min(load_keys), torch.max(load_keys)), file=log_f)

encoder_config = {}
decoder_config = {}
if args['encoder_type'] == 'partition':
    encoder_config['shifts'] = args['shifts']
    encoder_config['keep_res'] = args['keep_res']

if args['decoder_type'] == 'reduce':
    decoder_config['reduce_dim'] = args['reduce_dim']
elif args['decoder_type'] == 'shift_sum':
    decoder_config['shifts'] = args['shifts']
    decoder_config['keep_res'] = args['keep_res']
print('Encoder configs\t{}'.format(encoder_config))
print('Decoder configs\t{}'.format(decoder_config))
if args['log_file']:
    print('Encoder configs\t{}'.format(encoder_config), file=log_f)
    print('Decoder configs\t{}'.format(decoder_config), file=log_f)

# 训练集数据
num_train_keys = np.max((np.clip(int(load_keys.shape[0] * args['train_ratio']),
                                 1, load_keys.shape[0]), 10000))
train_keys_list = []
for i in range(args['num_train']):
    print('Process: Splitting training data...')
    print('Number of training keys {}'.format(num_train_keys))
    if args['log_file']:
        print('Number of training keys {}'.format(num_train_keys), file=log_f)
    train_keys = sample_data(load_keys, num_train_keys)
    train_keys_list.append(train_keys)

model = DistTransformer(args['num_flows'], args['num_layers'], args['input_dim'],
                        args['hidden_dim'], args['encoder_type'], encoder_config,
                        args['decoder_type'], decoder_config)
optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'],amsgrad=True)

print('Parameters={}, n_dims={}'.format(sum((p != 0).sum()
                                            if len(p.shape) > 1 else torch.tensor(p.shape).item()
                                            for p in model.parameters()), args['input_dim']))
print('{}'.format(model))

if args['log_file']:
    print('Parameters={}, n_dims={}'.format(sum((p != 0).sum()
                                                if len(p.shape) > 1 else torch.tensor(p.shape).item()
                                                for p in model.parameters()), args['input_dim']), file=log_f)
    oldstdout = sys.stdout
    sys.stdout = log_f
    print('{}'.format(model))
    sys.stdout = oldstdout

if args['load'] is not None:
    load_path = os.path.join(args['load'], 'checkpoint.pt')
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
else:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=args['decay'], patience=args['patience'], min_lr=5e-4,
                                                           verbose=True, threshold_mode='abs')
    best_loss = 1e20
    for i, train_keys in enumerate(train_keys_list):
        print('Process: {} Training...'.format(i + 1))
        start_time = time.time()
        best_loss = train(train_keys, model, optimizer, scheduler, best_loss, args, model_path)
        train_durations = time.time() - start_time
        print('Time Cost of Training {}'.format(train_durations))
        if args['log_file']:
            print('Time Cost of Training {}'.format(train_durations), file=log_f)
        load(model, optimizer, model_path)

print('Process: Saving weights for C++ inference...')
weight_path = os.path.join(checkpoint_dir, args['data_name'] + '-weights.txt')
save_weights(model, global_mean, global_var, args, weight_path)

print('Process: Transforming keys...')
tran_keys = test(load_keys, model, args)

num_unordered = ((tran_keys[1:] - tran_keys[:-1]) < 0).sum().item()
if num_unordered > 0:
    print('Process: Sorting keys...')
    tran_keys = torch.sort(tran_keys, 0)[0]

print('Process: Evaluating the transformed keys...')
evaluate_keys(tran_keys.detach().cpu().numpy().squeeze(), log_f=log_f)

if args['plot']:
    num_samples = 100000
    if load_keys.shape[0] > num_samples:
        print('Process: Sample data for plotting...')
        plot_keys = sample_data(load_keys, num_samples)
        plot_keys = torch.sort(plot_keys, 0)[0]
        tran_keys = test(plot_keys, model, args)
    print('Process: Plot the transformed distribution...')
    plot_dist(plot_keys.detach().cpu().numpy().squeeze(),
              os.path.join(checkpoint_dir, args['data_name'] + '_load_dist'), 'pdf')
    plot_dist(tran_keys.detach().cpu().numpy().squeeze(),
              os.path.join(checkpoint_dir, args['data_name'] + '_tran_dist'), 'pdf')
    plot_dist(plot_keys.detach().cpu().numpy().squeeze(),
              os.path.join(checkpoint_dir, args['data_name'] + '_load_dist'), 'cdf')
    plot_dist(tran_keys.detach().cpu().numpy().squeeze(),
              os.path.join(checkpoint_dir, args['data_name'] + '_tran_dist'), 'cdf')

transed_keys = test(load_keys, model, args)
transed_keys = np.sort(transed_keys.detach().cpu().numpy(), axis=0)

print(transed_keys)

plt.plot(transed_keys, [x for x in range(len(transed_keys))])

print(load_keys[1:2])

t = test(load_keys[1:2], model, args)

print(load_keys[1:2])
print(((patientunitstayid[1] - global_mean) / global_var).reshape(1, -1))

t[0].item() * load_keys.shape[0]

# 2 CDF切分

def slope(x1, x2, y1, y2):
    """两点斜率"""
    return (y2 - y1) / (x2 - x1)

def mutation(keys):
    slopes = []
    delta_slope = [np.array([0.0])]
    for i in range(len(keys) - 1):
        slopes.append(slope(keys[i], keys[i + 1], i, i + 1))
    slopes.append(np.array([0.0]))

    for j in range(1, len(slopes)):
        delta_slope.append(abs(slopes[j] - slopes[j - 1]))
    return delta_slope

mutations = np.array(mutation(transed_keys))
mutations

mutation_point1 = transed_keys[np.argmax(mutations)]
print(mutation_point1)

plt.plot(transed_keys, [x for x in range(len(transed_keys))])
plt.axvline(x=mutation_point1, color='r')
plt.show()
mutations[np.argmax(mutations)] = 0

mutation_point2 = transed_keys[np.argmax(mutations)]
print(mutation_point2)

mutations[np.argmax(mutations)] = 0

mutation_point3 = transed_keys[np.argmax(mutations)]
print(mutation_point3)
plt.plot(transed_keys, [x for x in range(len(transed_keys))])
plt.axvline(x=mutation_point1, color='r', label='1')
plt.axvline(x=mutation_point2, color='y', label='2')
plt.axvline(x=mutation_point3, color='g', label='3')
plt.legend()
plt.show()

# 递归划分和拟合线性模型

results = []
mutation_points = []
mutation_point_indexes = []

def cut_and_fit(keys, pos, mutations, low, up):
    if low + 1 == up:
        results.append([low, up, np.array([1.0]), np.array([1.0])])
        return

    model = LinearRegression()
    model.fit(keys[low:up], pos[low:up])
    r2 = model.score(keys[low:up], pos[low:up])

    if r2 >= 0.981481:
        results.append([low, up, model.coef_, model.intercept_, r2])
        return
    else:
        mutation_point_index = (np.argmax(mutations[low:up]).item() + low)
        mutation_point = keys[mutation_point_index]
        mutation_points.append(mutation_point)
        mutation_point_indexes.append(mutation_point_index)

        mutations[mutation_point_index] = np.array([0.0])
        if low < mutation_point_index:
            cut_and_fit(keys, pos, mutations, low, mutation_point_index)
        if mutation_point_index < up:
            cut_and_fit(keys, pos, mutations, mutation_point_index, up)

pos = np.arange(0, len(transed_keys)).reshape(-1, 1)
cut_and_fit(transed_keys, pos, mutations, 0, len(transed_keys) - 1)

print(pos)
print(len(results))
print(results)

mutation_points_copy = mutation_points
mutation_points.append(transed_keys[0])
mutation_points.append(transed_keys[-1])
mutation_points = np.sort(mutation_points, axis=0)

print(mutation_points)
print(len(mutation_points))

# 可视化分割结果
def gen_pre(keys, coef, intercept):
    return keys * coef + intercept

fig = plt.figure(figsize=(20, 8))
l = 0
u = -1
plt.scatter(transed_keys[l:u], pos[l:u], s=3)
for point in mutation_points:
    plt.axvline(x=point, color='r')
plt.show()

# 查询算法
def find_model(arr, x: torch.Tensor):
    arr = arr[arr[:, 0] <= x]
    arr = arr[arr[:, 1] > x]
    return arr[0]

def query(x: torch.Tensor):
    x_trans = (x - global_mean) / global_var
    x_trans = test(x_trans.reshape(-1, 1), model, args).item()
    model_params = find_model(np.array(results), x_trans)
    coef = model_params[2]
    intercept = model_params[3]
    y = x_trans * coef + intercept
    return y

# 示例查询
x = torch.Tensor([patientunitstayid[1]])
print(f"Query result for {x.item()}: {query(x)}")

# 关闭日志文件
if args['log_file']:
    log_f.close()

# 释放 SparkSession
spark.stop()
