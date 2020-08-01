import * as tf from "@tensorflow/tfjs";
const tens = tf.tensor([1, 2]);
console.log(tens);
window.onload = () => {
  document.body.innerHTML = JSON.stringify(tens);
}
