import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import {
  getIrisData,
  IRIS_CLASSES
} from './data';
import { callbacks } from '@tensorflow/tfjs';

window.onload = async () => {
  const [xTrain, yTrain, xTest, yTest] = getIrisData(0.15); // 其中参数0.15表示的意思是从数据集中分出来15%的数据作为验证集。xTrain为训练集的所有特征，yTrain为训练集的所有标签；xTest为训练集的所有特征，yTest为验证集的所有标签。
  console.log(xTrain.shape); // 输出[126,4]，表示训练集有126条数据
  console.log(xTest.shape); // 输出[24,4]，表示验证集有24条数据
  xTrain.print();
  yTrain.print();
  xTest.print();
  yTest.print();
  console.log(IRIS_CLASSES);

  // 初始化模型
  const model = tf.sequential();
  // 添加第一层
  model.add(tf.layers.dense({
    units: 10,
    inputShape: [xTrain.shape[1]],
    activation: 'sigmoid', // 这第一层的激活函数只要能带来非线性的变化就可以
  }));
  // 添加第二层
  model.add(tf.layers.dense({
    units: 3,
    activation: 'softmax'
  }));

  model.compile({
    loss: 'categoricalCrossentropy', // 交叉熵损失函数
    optimizer: tf.train.adam(0.1), // adam优化器
    metrics: ['accuracy'] // 这是准确度度量
  });

  // 训练模型
  await model.fit(xTrain, yTrain, {
    epochs: 100,
    validationData: [xTest, yTest], // 验证集
    callbacks: tfvis.show.fitCallbacks( // 可视化训练过程
      { name: '训练效果' },
      [
        'loss', // 训练集的损失
        'val_loss', // 验证集的损失
        'acc', // 训练集的准确度
        'val_acc' // 验证集的准确度
      ],
      { callbacks: ['onEpochEnd'] } // 设置只显示onEpochEnd，而不显示onBatchEnd
    )
  })

  window.predict = (form) => {
    const input = tf.tensor([[
      +form.a.value,
      +form.b.value,
      +form.c.value,
      +form.d.value,
    ]]); // 输入数据和训练数据的格式要保一致
    const pred = model.predict(input);
    alert(`预测结果：${IRIS_CLASSES[pred.argMax(1).dataSync(0)]}`); // pred.argMax(1)输出的是pred的第二维（0是第一维，1是第二维）的最大值的坐标，不过它输出的是一个Tensor，需要通过dataSync(0)转成普通数据进行显示
  }
}
