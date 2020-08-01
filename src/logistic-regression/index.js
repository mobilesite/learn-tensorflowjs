import * as tfvis from '@tensorflow/tfjs-vis';
import * as tf from '@tensorflow/tfjs';
import { getData } from './data.js';

window.onload = async () => {
  const data = getData(400); // 获取400个点
  console.log(data);

  tfvis.render.scatterplot(
    { name: '逻辑回归训练数据' },
    {
      values: [
        data.filter(p => p.label === 1),
        data.filter(p => p.label === 0),
      ] // 这里的数据跟之前的有些不同，这是一个二维数组
    }
  );

  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: 1, // 一个神经元
    inputShape: [2], // 特征长度为2（有两个特征）的1维数组
    activation: 'sigmoid' // 激活函数。之所以选择它的原因是它能保证输出结果在[0,1]之间
  }));

  model.compile(
    {
      loss: tf.losses.logLoss,
      optimizer: tf.train.adam(0.1)
    }
  );

  const inputs = tf.tensor(data.map((p) => [p.x, p.y]));
  const labels = tf.tensor(data.map((p) => p.label));

  await model.fit(inputs, labels, {
    batchSize: 40, // 因为整个训练数据是400个，这里设置了batchSize等于40，那就相当于10个batch就完成一轮训练，即10个batch构成一个epoch。
    epochs: 50,
    callbacks: tfvis.show.fitCallbacks(
      { name: '训练过程' },
      ['loss']
    )
  });

  window.predict = (form) => {
    // 预测
    const pred = model.predict(tf.tensor([[+form.x.value, +form.y.value]]));
    alert(`预测结果：${pred.dataSync()[0]}`)
  }
}