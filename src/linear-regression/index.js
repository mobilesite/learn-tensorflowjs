import * as tfvis from "@tensorflow/tfjs-vis";
import * as tf from "@tensorflow/tfjs";

window.onload = async () => {
  const xs = [1, 2, 3, 4];
  const ys = [1, 4, 7, 10];

  tfvis.render.scatterplot(
    { name: '线性回归训练集' },
    { values: xs.map((x, i) => ({x, y: ys[i]}))}, // 每个点的坐标
    { xAxisDomain: [0, 5], yAxisDomain: [0, 15]}, // x轴和y轴的显示区间
  );

  // 添加模型
  const model = tf.sequential(); // sequential方法会创建一个连续的模型，什么是连续的模型呢？就是这一层的输入一定是上一层的输出。

  // 添加层
  model.add(tf.layers.dense({
    units: 1, // 神经元的个数
    inputShape: [1], // inputShape是不允许写空数组的，[1]表示是一维的数据并且长度是1（即特征数量是1）
  }));

  // 设置损失函数
  model.compile({
    loss: tf.losses.meanSquaredError,
    optimizer: tf.train.sgd(0.1)
  }); // 设置损失函数和优化器

  // 训练数据转为Tensor
  const inputs = tf.tensor(xs); // 特征
  const labels = tf.tensor(ys); // 标签
  await model.fit(inputs, labels, {
    batchSize: 4, // 小批量随机梯度下降中的小批量的批量样本数
    epochs: 100, // 迭代整个训练数据的次数，这个也是个超参数，需要不断调整得到一个合适值
    callbacks: tfvis.show.fitCallbacks(
      {name: '训练过程'},
      ['loss'], // 度量单位，用于指定可视化想看什么，这是主要是想看损失情况
    )
  }); // 训练模型

  // 预测
  const output = model.predict(tf.tensor([5]));
  // 输出预测结果
  alert(`如果x为5，那么y为${output.dataSync()[0]}`);
}