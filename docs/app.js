let model;
const resultBox = document.getElementById("result");

async function loadModel() {
  resultBox.innerText = "Loading model...";
  model = await tf.loadLayersModel("model_tfjs/model.json");
  resultBox.innerText = "✅ Model ready! Enter a description and click Predict.";
}

loadModel();

async function predict() {
  const desc = document.getElementById("desc").value.trim();
  const cuisine = document.getElementById("cuisine").value;

  if (!desc) {
    resultBox.innerText = "Please enter a description.";
    return;
  }

  // В реальном проекте здесь должна быть та же предобработка (TF-IDF + SVD + OHE),
  // что и при обучении, но для демо просто берём длину текста и категорию.
  const textLength = desc.length / 100;
  const cuisineScore = cuisine ? cuisine.length / 10 : 0.5;

  const inputTensor = tf.tensor2d([[textLength, cuisineScore]]);
  const prediction = model.predict(inputTensor);
  const prob = (await prediction.data())[0];

  const quality = prob > 0.5 ? "Good pairing 👍" : "Poor pairing 👎";
  resultBox.innerText = `Predicted Quality: ${quality}  (confidence: ${(prob * 100).toFixed(1)}%)`;
}
