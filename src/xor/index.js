import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import {
  getData
} from './data';

window.onload = async () => {
  const data = getData(400);
  tfvis.render.scatterplot({
    name: 'XOR训练数据'
  }, {
    values: [
      data.filter(p => p.label === 1),
      data.filter(p => p.label === 0)
    ]
  });

  const model = tf.sequential(); // 初始化一个sequential模型
  model.add(tf.layers.dense({
    units: 4, // 神经元个数
    inputShape: [2], // 第一层需要设置inputShape，后面的层就不需要了，会自动设定（因为下一层的输入就是上一层的输出，比如，这里的这一层是4个神经元，那么下一层的inputShape其实可以自动得到，就是[4]，所以，下一层就不需要再指明inputShape了）。这里我们设置inputShape为2，即长度为2的一维数组，因为我们这里的特征个数是2。
    activation: 'relu', // 设一个激活函数，实现非线性的变化，我们这里任意选了一个非线性的relu
  }));
  // 接下来我们设置输出层
  model.add(tf.layers.dense({
    units: 1, // 输出层的神经元个数是1，因为我们只需要输出一个概率
    activation: 'sigmoid', // 因为输出层需要输出一个[0,1]之间的概率，所以这里activation只能设置成sigmoid
  }));
  model.compile({
    loss: tf.losses.logLoss, // 损失函数这里也用logLoss，因为它本质上也是一个逻辑回归
    optimizer: tf.train.adam(0.1), // 优化器使用adam，并且设置学习率为0.1
  });

  const inputs = tf.tensor(data.map(p => [p.x, p.y]));
  const labels = tf.tensor(data.map(p => p.label));

  // 进行训练
  await model.fit(inputs, labels, {
    epochs: 10,
    // 可视化训练过程
    callbacks: tfvis.show.fitCallbacks({
        name: '训练效果'
      }, // 图表的标题
      ['loss'] // 度量单位，只看损失
    )
  });

  window.predict = async (form) => {
    const pred = await model.predict(tf.tensor([
      [+form.x.value, +form.y.value]
    ]));
    alert(`预测结果：${pred.dataSync()[0]}`);
  };
}