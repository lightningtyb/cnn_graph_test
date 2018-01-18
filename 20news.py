
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

import sys, os
sys.path.insert(0, '..')
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# from lib import models, graph, coarsening, utils
import models,graph,coarsening,utils
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.sparse
import numpy as np
import time

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


flags = tf.app.flags
FLAGS = flags.FLAGS

# Graphs.
flags.DEFINE_integer('number_edges', 16, 'Graph: minimum number of edges per vertex.')
flags.DEFINE_string('metric', 'cosine', 'Graph: similarity measure (between features).')
# TODO: change cgcnn for combinatorial Laplacians.
flags.DEFINE_bool('normalized_laplacian', True, 'Graph Laplacian: normalized.')
flags.DEFINE_integer('coarsening_levels', 0, 'Number of coarsened graphs.')

flags.DEFINE_string('dir_data', os.path.join('..', 'data', '20news'), 'Directory to store data.')
flags.DEFINE_integer('val_size', 400, 'Size of the validation set.')


# # Data

# In[ ]:


# Fetch dataset. Scikit-learn already performs some cleaning.
remove = ('headers','footers','quotes')  # (), ('headers') or ('headers','footers','quotes')
cats=['alt.atheism','comp.graphics','comp.os.ms-windows.misc']
train = utils.Text20News(data_home=FLAGS.dir_data, subset='train', categories=cats,remove=remove)
#这个返回的train是什么呢

# Pre-processing: transform everything to a-z and whitespace.
# print('pre-processing',train.show_document(1)[:400])
pre_pro=train.show_document(1)[:400]
out1='out1--pre-processing'+str(pre_pro)+'\n\n'
with open('output1.txt', 'a+') as writter:
    writter.write(str(out1))
train.clean_text(num='substitute')

# Analyzing / tokenizing: transform documents to bags-of-words.
#stop_words = set(sklearn.feature_extraction.text.ENGLISH_STOP_WORDS)
# Or stop words from NLTK.
# Add e.g. don, ve.
train.vectorize(stop_words='english')#feature extraction and transform to vectors
# print(train.show_document(1)[:400])
analyzing=train.show_document(1)[:400]
print(analyzing)
out2="out2--analyzing--"+str(analyzing)+'\n\n'
with open('output1.txt', 'a+') as writter:
    writter.write(str(out2))


# In[ ]:


# Remove short documents.
train.data_info(True)#有out3,out4
out4_1='out4_1--data_info--Remove short documents.'+'\n\n'
with open('output1.txt', 'a+') as writter:
    writter.write(str(out4_1))

wc = train.remove_short_documents(nwords=20, vocab='full')
train.data_info()
print('shortest: {}, longest: {} words'.format(wc.min(), wc.max()))
out5='out5--data_info--shortest: '+str(wc.min())+'longest: '+str(wc.max())+'\n\n'
with open('output1.txt', 'a+') as writter:
    writter.write(str(out5))
plt.figure(figsize=(17,5))
plt.semilogy(wc, '.');

# Remove encoded images.
def remove_encoded_images(dataset, freq=1e3):
    widx = train.vocab.index('ax')
    wc = train.data[:,widx].toarray().squeeze()
    idx = np.argwhere(wc < freq).squeeze()
    dataset.keep_documents(idx)
    return wc
wc = remove_encoded_images(train)
train.data_info()
out5_1='out5_1--data_info--Remove encoded images'+'\n\n'
with open('output1.txt', 'a+') as writter:
    writter.write(str(out5_1))
plt.figure(figsize=(17,5))
plt.semilogy(wc, '.');#对y轴取对数


# In[ ]:


# Word embedding
if True:
    # """Embed the vocabulary using pre-trained vectors."""
    train.embed()
else:
    train.embed(os.path.join('..', 'data', 'word2vec', 'GoogleNews-vectors-negative300.bin'))
train.data_info()
out5_2='out5_2--data_info--Word embedding'+'\n\n'
with open('output1.txt', 'a+') as writter:
    writter.write(str(out5_2))
# Further feature selection. (TODO)


# In[ ]:


# Feature selection.
# Other options include: mutual information or document count.
out6_0='out6_0----Feature selection.\nOther options include: mutual information or document count.'+'\n\n'
with open('output1.txt', 'a+') as writter:
    writter.write(str(out6_0))
freq = train.keep_top_words(1000, 20)#out6,out7
train.data_info()
out6_1='out6_1--data_info--Feature selection.'+'\n\n'
with open('output1.txt', 'a+') as writter:
    writter.write(str(out6_1))
train.show_document(1)
plt.figure(figsize=(17,5))
plt.semilogy(freq);

# Remove documents whose signal would be the zero vector.
out8="out8--data_info--Remove documents whose signal would be the zero vector."+'\n\n'
with open('output1.txt', 'a+') as writter:
    writter.write(str(out8))
wc = train.remove_short_documents(nwords=5, vocab='selected')
train.data_info(True)


# In[ ]:

out9="out9--normalize"
with open('output1.txt', 'a+') as writter:
    writter.write(str(out9))
train.normalize(norm='l1')
train.show_document(1);


# In[ ]:


# Test dataset.
out10='out10--test dataset '
with open('output1.txt', 'a+') as writter:
    writter.write(str(out10))


cats=['alt.atheism','comp.graphics','comp.os.ms-windows.misc']
test = utils.Text20News(data_home=FLAGS.dir_data, subset='test',categories=cats, remove=remove)
test.clean_text(num='substitute')
test.vectorize(vocabulary=train.vocab)
test.data_info()
out10_1='out10_1--data_info--Test dataset..'+'\n\n'
with open('output1.txt', 'a+') as writter:
    writter.write(str(out10_1))
wc = test.remove_short_documents(nwords=5, vocab='selected')
print('shortest: {}, longest: {} words'.format(wc.min(), wc.max()))
out11='out11--data_info--shortest: '+str(wc.min())+'longest: '+str(wc.max())+'\n\n'
with open('output1.txt', 'a+') as writter:
    writter.write(str(out11))
test.data_info(True)
test.normalize(norm='l1')


# In[ ]:


if True:
    train_data = train.data.astype(np.float32)
    test_data = test.data.astype(np.float32)
    train_labels = train.labels
    test_labels = test.labels
    out12 = 'out12--if true :\n train_data='+str(train_data)
    out12+='\ntest_data='+str(test_data)
    out12+="\ntrain_labels"+str(train_labels)
    out12+="\ntest_labels"+str(test_labels)+'\n\n'
    with open('output1.txt', 'a+') as writter:
        writter.write(str(out12))
else:
    perm = np.random.RandomState(seed=42).permutation(dataset.data.shape[0])
    Ntest = 6695
    perm_test = perm[:Ntest]
    perm_train = perm[Ntest:]
    train_data = train.data[perm_train,:].astype(np.float32)
    test_data = train.data[perm_test,:].astype(np.float32)
    train_labels = train.labels[perm_train]
    test_labels = train.labels[perm_test]
    out12 = 'out12--if false :\n train_data='+ str(train_data)
    out12 += '\ntest_data='+ str(test_data)
    out12 += "\ntrain_labels"+str(train_labels)
    out12 += "\ntest_labels"+str(test_labels)+'\n\n'
    with open('output1.txt', 'a+') as writter:
        writter.write(str(out12))

if True:
    graph_data = train.embeddings.astype(np.float32)
else:
    graph_data = train.data.T.astype(np.float32).toarray()

out13 ="out13--graph_data="+str(graph_data)+'\n\n'
with open('output1.txt', 'a+') as writter:
    writter.write(str(out13))

#del train, test


# # Feature graph

# In[ ]:


t_start = time.process_time()
# flags = tf.app.flags
# FLAGS = flags.FLAGS
# flags.DEFINE_integer('number_edges', 16, 'Graph: minimum number of edges per vertex.')
# flags.DEFINE_string('metric', 'cosine', 'Graph: similarity measure (between features).')
# # TODO: change cgcnn for combinatorial Laplacians.
# flags.DEFINE_bool('normalized_laplacian', True, 'Graph Laplacian: normalized.')
# flags.DEFINE_integer('coarsening_levels', 0, 'Number of coarsened graphs.')
# 
# flags.DEFINE_string('dir_data', os.path.join('..', 'data', '20news'), 'Directory to store data.')
# flags.DEFINE_integer('val_size', 400, 'Size of the validation set.')

# """Compute exact pairwise distances.计算成对的距离"""
# graph_data
# k=4
# metric=euclidean
out14 ="out14--!!!!coarsening\n"
with open('output1.txt', 'a+') as writter:
    writter.write(out14)
print('coarsening1')
dist, idx = graph.distance_sklearn_metrics(graph_data, k=FLAGS.number_edges, metric=FLAGS.metric)
print('20news.py --dist={} ,idx={}'.format(dist,idx))
out15 ="out15--dist="+str(dist)+'idx='+str(idx)
out15+='\n'
with open('output1.txt', 'a+') as writter:
    writter.write(str(out15))


# 用ID和距离生成邻接矩阵knn算法实现
A = graph.adjacency(dist, idx)
print("邻接矩阵--{} > {} edges".format(A.nnz//2, FLAGS.number_edges*graph_data.shape[0]//2))
out16 ="out16--"+str(A.nnz//2)+'> '+str(FLAGS.number_edges*graph_data.shape[0]//2)
out16+='\n'
with open('output1.txt', 'a+') as writter:
    writter.write(str(out16))

# Replace randomly chosen edges by random edges."""参数是（邻接矩阵，noise_level）
A = graph.replace_random_edges(A, 0)
out17 ="out17--replace_random_edges(A, 0)\n"
out17+='\n'
with open('output1.txt', 'a+') as writter:
    writter.write(str(out17))

#A是邻接矩阵，levels粗化了的图的数目（0），self_connection=false
print('\n\ncoarsening\n\n')
graphs, perm = coarsening.coarsen(A, levels=FLAGS.coarsening_levels, self_connections=False)
print('graphs = {},perm = {}'.format(graphs,perm))
out18 ="out18--!!!!real coarsening\n"
out18+="graphs="+str(graphs)
out18+='\nperm='+str(perm)+'\n'
out18+='\n'
with open('output1.txt', 'a+') as writter:
    writter.write(str(out18))

    # """Return the Laplacian of the weigth matrix."""
L = [graph.laplacian(A, normalized=True) for A in graphs]
print("L={}".format(L))
out19 ="out19--Return the Laplacian of the weigth matrix.\n"
out19+='L= '+str(L)+'\n'
out19+='\n'
with open('output1.txt', 'a+') as writter:
    writter.write(str(out19))

exe_time=time.process_time() - t_start
# print('Execution time: {:.2f}s'.format(time.process_time() - t_start))
print('Execution time: {:.2f}s'.format(exe_time))
out20="out20--Execution time:"+str(exe_time)+'\n'
out20+='\n'
with open('output1.txt', 'a+') as writter:
    writter.write(str(out20))
#graph.plot_spectrum(L)
#del graph_data, A, dist, idx


# In[ ]:


t_start = time.process_time()


train_data = scipy.sparse.csr_matrix(coarsening.perm_data(train_data.toarray(), perm))
print('train_data--',train_data)
test_data = scipy.sparse.csr_matrix(coarsening.perm_data(test_data.toarray(), perm))
print('test_data--',test_data)
print('Execution time: {:.2f}s'.format(time.process_time() - t_start))
del perm


# # Classification
# 
# **Reminder**: change the optimizer to Adam in `lib/models.py`.

# In[ ]:

# Validation set.
if False:
    val_data = train_data[:FLAGS.val_size,:]
    val_labels = train_labels[:FLAGS.val_size]
    train_data = train_data[FLAGS.val_size:,:]
    train_labels = train_labels[FLAGS.val_size:]
else:
    val_data = test_data
    val_labels = test_labels


# In[ ]:


if True:
    utils.baseline(train_data, train_labels, test_data, test_labels)


# In[ ]:


common = {}
common['dir_name']       = '20news/'
common['num_epochs']     = 80
common['batch_size']     = 100
common['decay_steps']    = len(train_labels) / common['batch_size']
common['eval_frequency'] = 5 * common['num_epochs']
common['filter']         = 'chebyshev5'
common['brelu']          = 'b1relu'
common['pool']           = 'mpool1'
C = max(train_labels) + 1  # number of classes

model_perf = utils.model_perf()


# In[ ]:


if True:
    name = 'softmax'
    params = common.copy()
    params['dir_name'] += name
    params['regularization'] = 0
    params['dropout']        = 1
    params['learning_rate']  = 1e3
    params['decay_rate']     = 0.95
    params['momentum']       = 0.9
    params['F']              = []
    params['K']              = []
    params['p']              = []
    params['M']              = [C]
    model_perf.test(models.cgcnn(L, **params), name, params,
                    train_data, train_labels, val_data, val_labels, test_data, test_labels)


# In[ ]:


if True:
    name = 'fc_softmax'
    params = common.copy()
    params['dir_name'] += name
    params['regularization'] = 0
    params['dropout']        = 1
    params['learning_rate']  = 0.1
    params['decay_rate']     = 0.95
    params['momentum']       = 0.9
    params['F']              = []
    params['K']              = []
    params['p']              = []
    params['M']              = [2500, C]
    model_perf.test(models.cgcnn(L, **params), name, params,
                    train_data, train_labels, val_data, val_labels, test_data, test_labels)


# In[ ]:


if True:
    name = 'fc_fc_softmax'
    params = common.copy()
    params['dir_name'] += name
    params['regularization'] = 0
    params['dropout']        = 1
    params['learning_rate']  = 0.1
    params['decay_rate']     = 0.95
    params['momentum']       = 0.9
    params['F']              = []
    params['K']              = []
    params['p']              = []
    params['M']              = [2500, 500, C]
    model_perf.test(models.cgcnn(L, **params), name, params,
                    train_data, train_labels, val_data, val_labels, test_data, test_labels)


# In[ ]:


if True:
    name = 'fgconv_softmax'
    params = common.copy()
    params['dir_name'] += name
    params['filter']         = 'fourier'
    params['regularization'] = 0
    params['dropout']        = 1
    params['learning_rate']  = 0.001
    params['decay_rate']     = 1
    params['momentum']       = 0
    params['F']              = [32]
    params['K']              = [L[0].shape[0]]
    params['p']              = [1]
    params['M']              = [C]
    model_perf.test(models.cgcnn(L, **params), name, params,
                    train_data, train_labels, val_data, val_labels, test_data, test_labels)


# In[ ]:


if True:
    name = 'sgconv_softmax'
    params = common.copy()
    params['dir_name'] += name
    params['filter']         = 'spline'
    params['regularization'] = 1e-3
    params['dropout']        = 1
    params['learning_rate']  = 0.1
    params['decay_rate']     = 0.999
    params['momentum']       = 0
    params['F']              = [32]
    params['K']              = [5]
    params['p']              = [1]
    params['M']              = [C]
    model_perf.test(models.cgcnn(L, **params), name, params,
                    train_data, train_labels, val_data, val_labels, test_data, test_labels)


# In[ ]:


if True:
    name = 'cgconv_softmax'
    params = common.copy()
    params['dir_name'] += name
    params['regularization'] = 1e-3
    params['dropout']        = 1
    params['learning_rate']  = 0.1
    params['decay_rate']     = 0.999
    params['momentum']       = 0
    params['F']              = [32]
    params['K']              = [5]
    params['p']              = [1]
    params['M']              = [C]
    model_perf.test(models.cgcnn(L, **params), name, params,
                    train_data, train_labels, val_data, val_labels, test_data, test_labels)


# In[ ]:


if True:
    name = 'cgconv_fc_softmax'
    params = common.copy()
    params['dir_name'] += name
    params['regularization'] = 0
    params['dropout']        = 1
    params['learning_rate']  = 0.1
    params['decay_rate']     = 0.999
    params['momentum']       = 0
    params['F']              = [5]
    params['K']              = [15]
    params['p']              = [1]
    params['M']              = [100, C]
    model_perf.test(models.cgcnn(L, **params), name, params,
                    train_data, train_labels, val_data, val_labels, test_data, test_labels)


# In[ ]:


model_perf.show()


# In[ ]:


if False:
    grid_params = {}
    data = (train_data, train_labels, val_data, val_labels, test_data, test_labels)
    utils.grid_search(params, grid_params, *data, model=lambda x: models.cgcnn(L,**x))

