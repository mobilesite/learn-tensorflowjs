import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { MnistData } from './data';

window.onload = async () => {
  const data = new MnistData();
  await data.load();
  const examples = data.nextTestBatch(20); // nextTestBatch方法用于加载一些验证集，其参数是加载验证集的个数
  console.log(examples);

  const surface = tfvis.visor().surface({ name: '输入示例' });
  for(let i = 0; i < 20; i++) {
    const imageTensor = tf.tidy(() => {
      return examples.xs.slice([i, 0], [1, 784])
        .reshape([28, 28, 1]);
    });
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style.margin = '4px';
    await tf.browser.toPixels(imageTensor, canvas);
    // document.body.appendChild(canvas);
    surface.drawArea.appendChild(canvas);
  }

  const model = tf.sequential();

  /**
   * 先进行第一轮特征提取
   */
  // 卷积层
  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));
  // 最大池化层
  model.add(tf.layers.maxPool2d({
    poolSize: [2, 2],
    strides: [2, 2]
  }));
  /**
   * 再进行第二轮特征提取
   */
  // 卷积层
  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));
  // 最大池化层
  model.add(tf.layers.maxPool2d({
    poolSize: [2, 2],
    strides: [2, 2]
  }));
  // 把高维的特征图转化成一维的
  model.add(tf.layers.flatten());
  // 全链接层
  model.add(tf.layers.dense({
    units: 10,
    activation: 'softmax',
    kernelInitializer: 'varianceScaling'
  }));

  // 设置损失函数和优化器
  model.compile({
    loss: 'categoricalCrossentropy', // 交叉熵
    optimizer: tf.train.adam(), // 使用adam优化器
    metrics: ['accuracy'] // 可以看到准确度
  });

  // 准备训练集
  // 把Tensor操作放在tidy里面，这样中间的Tensor就会被清除掉，从而不会驻留在内存中影响性能
  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(1000);
    console.log(d); // 发现是shape [1000, 784]，而上面我们定义的模型需要的数据shape是[28,28,1]，所以需要reshape一下
    return [
      d.xs.reshape([1000, 28, 28, 1]), // 1000张图片，28 * 28像素的，灰度图片
      d.labels
    ];
  });

  // 准备验证集
  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(200);
    return [
      d.xs.reshape([200, 28, 28, 1]), // 用200张图片
      d.labels
    ];
  });

  // 训练模型
  await model.fit(trainXs, trainYs, {
    validationData: [testXs, testYs],
    batchSize: 500,
    epochs: 50,
    callbacks: tfvis.show.fitCallbacks({
        name: '训练效果'
      },
      ['loss', 'val_loss', 'acc', 'val_acc'], {
        callbacks: ['onEpochEnd']
      }
    )
  });

  /**
   * 实现画布的清除、初始化以及画布上输入手写数字的功能
  */
  const canvas = document.querySelector('canvas');
  canvas.addEventListener('mousemove', (e) => {
    if (e.buttons === 1) {
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = 'rgba(255, 255, 255)';
      ctx.fillRect(e.offsetX, e.offsetY, 25, 25);
    }
  })
  window.clear = () => {
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'rgba(0,0,0)';
    // 给整个画布铺上黑底
    ctx.fillRect(0, 0, 300, 300);
  }

  // 初始化的时候也给铺上黑底
  clear();

  window.predict = () => {
    // 进行数据处理
    const input = tf.tidy(() => {
      /**
       * 这里，我们需要进行一系列的数据处理动作。
       * 首先， 需要将Canvas对象转成Tensor。 Tensorflow给我们提供了把Canvas对象转成Tesnsor的方法： `tf.browser.fromPixels(canvas)`。
       其次， 然后我们需要将图的大小进行转化。 可以通过 tf.image.resizeBilinear方法， 其第一个参数是待转换的Tensor， 第二个参数是要转成的目标大小（ 这里是28 * 28 像素， 第三个参数是AlignCorners， 我们设置为true）。
       第三， 我们还要将彩色图片变成黑白图片。 我们用.slice([0, 0, 0], [28, 28, 1])。
       第四， 还需要对数据进行归一化。 用.toFloat().div(255)。
       最后， 这个shape要reshape成模型所需要的shape。
       通过 reshape([1, 28, 28, 1]) 完成。
       */
      return tf.image.resizeBilinear(
        tf.browser.fromPixels(canvas),
        [28, 28],
        true
      )
      .slice([0, 0, 0], [28, 28, 1])
      .toFloat().div(255)
      .reshape([1, 28, 28, 1]);
    });
    // 进行预测，.argMax(1)是拿到最大的那个值
    const pred = model.predict(input).argMax(1);
    alert(`预测结果为${pred.dataSync()[0]}`);
  }
}