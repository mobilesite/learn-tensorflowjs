import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
// import {getData} from './data';
import {getData} from '../xor/data';

window.onload = async () => {
  // const data = getData(200, 3);
  const data = getData(200);

  console.log(data);
  tfvis.render.scatterplot(
    { name: '训练数据' },
    {
      values: [
        data.filter(p => p.label === 1),
        data.filter(p => p.label === 0)
      ]
    }
  );

  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: 1,
    activation: 'sigmoid',
    inputShape: [2]
  }));
  model.compile({
    loss: tf.losses.logLoss,
    optimizer: tf.train.adam(0.1)
  });

  const inputs = tf.tensor(data.map(p => [p.x, p.y]));
  const labels = tf.tensor(data.map(p => p.label));

  await model.fit(inputs, labels, {
    validationSplit: 0.2, // 从数据集里面分出20%的数据作为验证集
    epochs: 200,
    callbacks: tfvis.show.fitCallbacks(
      { name: '训练效果' },
      ['loss', 'val_loss'], // 要看到训练集和验证集上的损失
      { callbacks: ['onEpochEnd']}
    )
  });
}