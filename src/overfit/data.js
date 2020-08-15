/**
 *
 * @param {*} numSamples 生成的样本的数量
 * @param {*} variance 方差，变异，不一样的地方。它是用来控制生成的数据的噪音的，variance调得越大，生成的数据的噪音就越大
 */
export function getData(numSamples, variance) {
  let points = [];

  function genGauss(cx, cy, label) {
    for (let i = 0; i < numSamples / 2; i++) {
      let x = normalRandom(cx, variance);
      let y = normalRandom(cy, variance);
      points.push({x, y, label});
    }
  }

  genGauss(2, 2, 1);
  genGauss(-2, -2, 0);
  return points;
}

/**
 * 生成一个正态分布，也叫高斯分布
 * @param {*} mean
 * @param {*} variance
 */
function normalRandom(mean = 0, variance = 1) {
  let v1, v2, s;
  do {
    v1 = 2 * Math.random() - 1;
    v2 = 2 * Math.random() - 1;
    s = v1 * v1 + v2 * v2;
  } while (s > 1);

  let result = Math.sqrt(-2 * Math.log(s) / s) * v1;
  return mean + Math.sqrt(variance) * result;
}