import tensorflow as tf
import numpy as np
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

print(tf.test.gpu_device_name())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

path_to_file = "../Three Kingdoms.txt"
print(f'pwd: {os.getcwd()}')
print(f'file path: {path_to_file}')

# 读取并为 py2 compat 解码
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# 文本长度是指文本中的字符个数
print ('Length of text: {} characters'.format(len(text)))

# 看一看文本中的前 250 个字符
print(text[:250])

# 文本中的非重复字符
vocab = sorted(set(text))
print ('{} unique characters'.format(len(vocab)))

# 创建从非重复字符到索引的映射
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')

# 显示文本首 13 个字符的整数映射
print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

# 设定每个输入句子长度的最大值
seq_length = 100
examples_per_epoch = len(text)//seq_length

# 创建训练样本 / 目标
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
  print(idx2char[i.numpy()])

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

#`batch` 方法使我们能轻松把单个字符转换为所需长度的序列。
for item in sequences.take(5):
  print(repr(''.join(idx2char[item.numpy()])))

#对于每个序列，使用 `map` 方法先复制再顺移，以创建输入文本和目标文本。`map` 方法可以将一个简单的函数应用到每一个批次 （batch）。
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))

for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

# 批大小
BATCH_SIZE = 8

# 设定缓冲区大小，以重新排列数据集
# （TF 数据被设计为可以处理可能是无限的序列，
# 所以它不会试图在内存中重新排列整个序列。相反，
# 它维持一个缓冲区，在缓冲区重新排列元素。） 
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# 词集的长度
vocab_size = len(vocab)

# 嵌入的维度
embedding_dim = 256

# RNN 的单元数量
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)

# for input_example_batch, target_example_batch in dataset.take(1):
#   example_batch_predictions = model(input_example_batch)
#   print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

model.summary()

# sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
# sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()

# print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
# print()
# print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# example_batch_loss  = loss(target_example_batch, example_batch_predictions)
# print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
# print("scalar_loss:      ", example_batch_loss.numpy().mean())

model.compile(optimizer='adam', loss=loss)

# 检查点保存至的目录
checkpoint_dir = './training_checkpoints_2'

# 检查点的文件名
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS=10

# history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

# tf.train.latest_checkpoint(checkpoint_dir)

# model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

# model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

# model.build(tf.TensorShape([1, None]))

# model.summary()

def generate_text(model, start_string):
  # 评估步骤（用学习过的模型生成文本）

  # 要生成的字符个数
  num_generate = 1000

  # 将起始字符串转换为数字（向量化）
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # 空字符串用于存储结果
  text_generated = []

  # 低温度会生成更可预测的文本
  # 较高温度会生成更令人惊讶的文本
  # 可以通过试验以找到最好的设定
  temperature = 1.0

  # 这里批大小为 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # 删除批次的维度
      predictions = tf.squeeze(predictions, 0)

      # 用分类分布预测模型返回的字符
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # 把预测字符和前面的隐藏状态一起传递给模型作为下一个输入
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

# print(generate_text(model, start_string=u"三國 "))

#--
## 高级：自定义训练
# 上面的训练步骤简单，但是能控制的地方不多。
# 至此，你已经知道如何手动运行模型。现在，让我们打开训练循环，并自己实现它。这是一些任务的起点，例如实现 _课程学习_ 以帮助稳定模型的开环输出。
# 你将使用 `tf.GradientTape` 跟踪梯度。关于此方法的更多信息请参阅 [eager execution 指南](https://tensorflow.google.cn/guide/eager)。

# 步骤如下：
# * 首先，初始化 RNN 状态，使用 `tf.keras.Model.reset_states` 方法。
# * 然后，迭代数据集（逐批次）并计算每次迭代对应的 *预测*。
# * 打开一个 `tf.GradientTape` 并计算该上下文时的预测和损失。
# * 使用 `tf.GradientTape.grads` 方法，计算当前模型变量情况下的损失梯度。
# * 最后，使用优化器的 `tf.train.Optimizer.apply_gradients` 方法向下迈出一步。

model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(inp, target):
  with tf.GradientTape() as tape:
    predictions = model(inp)
    loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            target, predictions, from_logits=True))
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  return loss

# 训练步骤
EPOCHS = 50

for epoch in range(EPOCHS):
  start = time.time()

  # 在每个训练周期开始时，初始化隐藏状态
  # 隐藏状态最初为 None
  hidden = model.reset_states()

  for (batch_n, (inp, target)) in enumerate(dataset):
    loss = train_step(inp, target)

    if batch_n % 100 == 0:
      template = 'Epoch {} Batch {} Loss {}'
      print(template.format(epoch+1, batch_n, loss))

  # 每 5 个训练周期，保存（检查点）1 次模型
  if (epoch + 1) % 5 == 0:
    model.save_weights(checkpoint_prefix.format(epoch=epoch))

  print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
  print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

model.save_weights(checkpoint_prefix.format(epoch=epoch))
model.summary()
tf.train.latest_checkpoint(checkpoint_dir)

model_2 = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model_2.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model_2.build(tf.TensorShape([1, None]))
model_2.summary()
print(generate_text(model_2, start_string=u"三國 "))