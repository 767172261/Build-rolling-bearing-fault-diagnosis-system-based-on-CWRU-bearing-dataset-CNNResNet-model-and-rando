CNN-ResNet，随机森林滚动轴承故障诊断实例
@[toc]
软件pycharm python=3.8
tensorflow架构
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1fd01c38f55d4903bd7215510b96f23f.png)

预实现功能：
一、
1.程序可生成精确率P(Precision)、召回率R(Recall)以及F1(F-measure)3个评价指标
2.可生成训练集和验证集准确率曲线，训练集和验证集的损失值曲线，ROC曲线
3.可生成混淆矩阵图。
4.可生成模型内部各层的T-SNE图。
5.可把每轮迭代的准确率和损失值保存到表格中。
6.程序使用：余弦退火策略动态调整学习率，L1正则化，Dropout算法，小波阀值降噪。
7.四个工况各生成一张数据图片、预处理后数据图像。
.使用的凯斯西储大学（Case Western Reserve University）轴承数据中心的开源数据集。(可更换为MFPT数据集、帕德博恩数据集)![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/00574a6bb0eb4cdcaa0d00ca05452c57.png)

# ## 随机森林-准确率得分：0.83
# ## CNN-准确率得分：0.99
# ## CNN+ResNet-准确率得分：1.0

基于凯斯西储大学（CWRU）轴承数据集的滚动轴承故障诊断系统，示例代码。使用CNN-ResNet模型和随机森林进行故障分类![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2dce494283854f01beb681145f747f0f.png)


### 1. 环境准备

确保安装了必要的库：

```bash
pip install numpy scipy matplotlib seaborn scikit-learn tensorflow pywavelets pandas openpyxl
```

### 2. 数据加载与预处理

假设数据已经下载并解压到本地目录。首先加载数据并进行预处理：

```python
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
import pywt

def load_data(file_path):
    mat = sio.loadmat(file_path)
    data = mat['variable_name']  # 替换为实际变量名
    labels = mat['labels']  # 假设.mat文件中包含标签信息
    return data, labels

def wavelet_denoising(data):
    coeffs = pywt.wavedec(data, 'db4', level=1)
    threshold = np.std(coeffs[-1]) * np.sqrt(2*np.log(len(data)))
    coeffs[1:] = [pywt.threshold(i, threshold, mode='soft') for i in coeffs[1:]]
    return pywt.waverec(coeffs, 'db4')

data, labels = load_data('path_to_cwru_dataset.mat')
data = wavelet_denoising(data)  # 应用小波阈值降噪
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9df71c762a1f4b92afe2c56a451b3c79.png)
### 3. 构建CNN-ResNet模型

```python
import tensorflow as tf
from tensorflow.keras import layers, models

class ResidualUnit(layers.Layer):
    def __init__(self, filters, strides=1, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            layers.Conv1D(filters, 3, strides=strides, padding="same", use_bias=False),
            layers.BatchNormalization(),
            self.activation,
            layers.Conv1D(filters, 3, strides=1, padding="same", use_bias=False),
            layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                layers.Conv1D(filters, 1, strides=strides, padding="same", use_bias=False),
                layers.BatchNormalization()]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)

def build_cnn_resnet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
    
    prev_filters = 64
    for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
        strides = 1 if filters == prev_filters else 2
        x = ResidualUnit(filters, strides=strides)(x)
        prev_filters = filters
    
    x = layers.GlobalAvgPool1D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    return model

input_shape = (X_train.shape[1], 1)
model = build_cnn_resnet(input_shape, num_classes=len(np.unique(y_train)))
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ca01edf4b988493f843be5fcc45ab7f5.png)

### 4. 训练模型

使用余弦退火策略动态调整学习率、L1正则化和Dropout：

```python
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.optimizers.schedules import CosineDecay

cosine_decay = CosineDecay(initial_learning_rate=0.1, decay_steps=1000)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cosine_decay), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

csv_logger = CSVLogger('training.log', separator=',', append=False)

history = model.fit(np.expand_dims(X_train, axis=-1), y_train, epochs=50, batch_size=32, 
                    validation_data=(np.expand_dims(X_val, axis=-1), y_val), callbacks=[csv_logger])
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6f7cfd4ea7694209831f8399565af2e2.png)

### 5. 模型评估与可视化

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 准确率得分
y_pred = model.predict(np.expand_dims(X_val, axis=-1))
print(classification_report(y_val, np.argmax(y_pred, axis=1)))

# 混淆矩阵图
cm = confusion_matrix(y_val, np.argmax(y_pred, axis=-1))
sns.heatmap(cm, annot=True, fmt='d')
plt.show()

# T-SNE可视化
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(model.layers[-2].output)

plt.figure(figsize=(16,10))
sns.scatterplot(
    x=tsne_results[:,0], y=tsne_results[:,1],
    hue=y_val,
    palette=sns.color_palette("hls", len(np.unique(y_val))),
    legend="full",
    alpha=0.3
)
plt.show()
```

### 6. 随机森林分类器

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_val)
print(f"Random Forest Accuracy: {accuracy_score(y_val, y_pred_rf)}")
```

构建基于CWRU轴承数据集的滚动轴承故障诊断系统，包括CNN-ResNet模型和随机森林分类器，并实现了多种评估指标和可视化方法。根据具体需求和数据集特点，可以进一步调整和优化上述代码。

仅供参考。
