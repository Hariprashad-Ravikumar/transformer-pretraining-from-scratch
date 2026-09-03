// Runs the model entirely in the browser: ONNX Runtime Web for inference,
// Transformers.js's AutoTokenizer (only) for tokenization. The sampling loop
// below is a JS port of the model's own generate_simple() -- same
// temperature/top-k/repetition-penalty logic, same fixed-length-recompute-
// each-step strategy (no KV cache), so behavior matches the Python
// reference exactly.

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/+esm";
import { AutoTokenizer } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/+esm";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/";

const MODEL_REPO = "hari-8/transformer-pretraining-from-scratch";
const BLOCK_SIZE = 1024;
const VOCAB_SIZE = 16384;
const PAD_ID = 0; // <|endoftext|> -- causally masked out, never affects real-token logits
const REPETITION_PENALTY = 1.3; // Keskar et al. 2019 -- see generate_simple() in the Python model

const statusEl = document.getElementById("status");
const outputEl = document.getElementById("output");
const generateBtn = document.getElementById("generateBtn");

let session = null;
let tokenizer = null;

function setStatus(text) {
  statusEl.textContent = text;
}

async function loadModel() {
  setStatus("Loading tokenizer...");
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_REPO);

  setStatus("Loading model (~344MB, cached after first load)...");
  session = await ort.InferenceSession.create("./model.onnx", {
    executionProviders: ["webgpu", "wasm"],
  });

  setStatus("Ready.");
  generateBtn.disabled = false;
}

function sampleFromLogits(logits, temperature, topK, seenIds) {
  // logits: Float32Array of length VOCAB_SIZE
  const scaled = new Float32Array(VOCAB_SIZE);
  for (let i = 0; i < VOCAB_SIZE; i++) scaled[i] = logits[i] / temperature;

  // repetition penalty: soften logits of tokens already generated, so a
  // high-probability token can't repeat indefinitely ("voltage-voltage-
  // voltage..."), a real failure mode of this small a model without it.
  if (REPETITION_PENALTY && REPETITION_PENALTY !== 1.0) {
    for (const id of seenIds) {
      const v = scaled[id];
      scaled[id] = v > 0 ? v / REPETITION_PENALTY : v * REPETITION_PENALTY;
    }
  }

  // top-k: find the k-th largest value, mask everything below it
  const indexed = Array.from(scaled, (v, i) => [v, i]);
  indexed.sort((a, b) => b[0] - a[0]);
  const threshold = indexed[Math.min(topK, indexed.length) - 1][0];
  for (let i = 0; i < VOCAB_SIZE; i++) {
    if (scaled[i] < threshold) scaled[i] = -Infinity;
  }

  // softmax
  let max = -Infinity;
  for (let i = 0; i < VOCAB_SIZE; i++) if (scaled[i] > max) max = scaled[i];
  let sum = 0;
  const probs = new Float32Array(VOCAB_SIZE);
  for (let i = 0; i < VOCAB_SIZE; i++) {
    const p = scaled[i] === -Infinity ? 0 : Math.exp(scaled[i] - max);
    probs[i] = p;
    sum += p;
  }
  for (let i = 0; i < VOCAB_SIZE; i++) probs[i] /= sum;

  // multinomial sample
  let r = Math.random();
  for (let i = 0; i < VOCAB_SIZE; i++) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return VOCAB_SIZE - 1;
}

async function generate(promptText, maxNewTokens, temperature, topK) {
  const encoded = await tokenizer(promptText, { return_tensor: false });
  let ids = Array.from(encoded.input_ids);

  for (let step = 0; step < maxNewTokens; step++) {
    if (ids.length >= BLOCK_SIZE) break;

    const padded = new BigInt64Array(BLOCK_SIZE);
    for (let i = 0; i < BLOCK_SIZE; i++) {
      padded[i] = BigInt(i < ids.length ? ids[i] : PAD_ID);
    }
    const inputTensor = new ort.Tensor("int64", padded, [1, BLOCK_SIZE]);

    const results = await session.run({ input_ids: inputTensor });
    const logitsData = results.logits.data; // Float32Array, shape [1, BLOCK_SIZE, VOCAB_SIZE]

    const lastPos = ids.length - 1;
    const offset = lastPos * VOCAB_SIZE;
    const nextTokenLogits = logitsData.subarray(offset, offset + VOCAB_SIZE);

    const nextId = sampleFromLogits(nextTokenLogits, temperature, topK, new Set(ids));
    ids.push(nextId);

    const decoded = await tokenizer.decode(ids, { skip_special_tokens: true });
    outputEl.textContent = decoded;
    setStatus(`Generating... (${step + 1}/${maxNewTokens})`);
  }

  setStatus("Done.");
}

generateBtn.addEventListener("click", async () => {
  generateBtn.disabled = true;
  outputEl.textContent = "";
  const prompt = document.getElementById("prompt").value;
  const maxTokens = parseInt(document.getElementById("maxTokens").value, 10);
  const temperature = parseFloat(document.getElementById("temperature").value);
  const topK = parseInt(document.getElementById("topK").value, 10);

  if (!prompt.trim()) {
    outputEl.textContent = "Enter a prompt first.";
    generateBtn.disabled = false;
    return;
  }

  try {
    await generate(prompt, maxTokens, temperature, topK);
  } catch (err) {
    setStatus("Error: " + err.message);
    console.error(err);
  } finally {
    generateBtn.disabled = false;
  }
});

generateBtn.disabled = true;
loadModel().catch((err) => {
  setStatus("Failed to load model: " + err.message);
  console.error(err);
});
