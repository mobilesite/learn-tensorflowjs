import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

window.onload = async () => {
  const heights = [150, 160, 170];
  const weights = [40, 50, 60];

  tfvis.render.scatterplot(
    { name: '身高体重训练数据' },
    { values: heights.map((x, i) => ({ x, y: weights[i] }))},
    {
      xAxisDomain: [140, 180],
      yAxisDomain: [30, 70]
    }
  );

  const inputs = tf.tensor(heights).sub(150).div(20);
  inputs.print(); // 会log出来[0, 0.5, 1]
  const labels = tf.tensor(weights).sub(40).div(20);
  labels.print(); // 会log出来[0, 0.5, 1]

  // 添加模型
  const model = tf.sequential(); // sequential方法会创建一个连续的模型，什么是连续的模型呢？就是这一层的输入一定是上一层的输出。
  // 其中tf.layers.dense()会生成一个全链接层。该层实现了如下操作：outputs = activation(dot(input, kernel) + bias)，其中activation是作为activation参数传递的激活函数，input是输入，kernel是由层创建的权重矩阵，bias是由层创建的偏差向量（偏置）。
  model.add(tf.layers.dense({
    units: 1, // 神经元的个数
    inputShape: [1] // inputShape是不允许写空数组的，[1]表示是一维的数据并且长度是1（即特征数量是1）
  }));
  model.compile({
    loss: tf.losses.meanSquaredError, // 损失函数：均方误差
    optimizer: tf.train.sgd(0.1) // 优化器：随机梯度下降，括号内的参数为学习速率
  })

  await model.fit(inputs, labels, {
    batchSize: 3, // 小批量随机梯度下降中的小批量的批量样本数
    epochs: 200, // 迭代整个训练数据的次数，这个也是个超参数，需要不断调整得到一个合适值
    callbacks: tfvis.show.fitCallbacks({
        name: '训练过程'
      },
      ['loss'], // 度量单位，用于指定可视化想看什么，这是主要是想看损失情况
    )
  }); // 训练模型

  // 预测
  const output = model.predict(tf.tensor([180]).sub(150).div(20)); // 注意这里传入的参数需要归一化
  // 输出预测结果
  alert(`如果身高为180cm，那么预测体重为${output.mul(20).add(40).dataSync()[0]}kg`); // 注意这里需要做反归一化
}