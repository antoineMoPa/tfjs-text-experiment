import { readFileSync, existsSync, createWriteStream, createReadStream } from 'fs';
import type { Tensor2D } from '@tensorflow/tfjs-core/dist/tensor';
import type { Sequential } from '@tensorflow/tfjs-layers/dist/models';
import type { UniversalSentenceEncoder } from  '@tensorflow-models/universal-sentence-encoder';
import * as tf from '@tensorflow/tfjs-node-gpu';

import * as _ from 'lodash';
import { performance } from 'perf_hooks';
import { LayersModel } from '@tensorflow/tfjs-node-gpu';
import * as json from 'big-json';
import * as assert from 'node:assert';
import * as WordPOS from 'wordpos';

const EMBED_SHAPE = [1, 512];
const BEFORE_SIZE = 5;
const CORPUS_PATH = "data/data-corpus.txt";
const scentenceEncoderModel: UniversalSentenceEncoder = global.universalSentenceEncoderModel;
const WORD_PREDICT_MODEL_CACHE = 'file://data/wordPredictModel';
const wordpos = new WordPOS({ stopwords: false });

type Corpus = {
    wordTensors: Tensor2D;
    words: string[];
};

async function buildCorpus() {
    console.log('Building corpus');
    const t1 = performance.now();
    const text = readFileSync(CORPUS_PATH).toString();
    const words: string[] = _.uniq(wordpos.parse(text));
    const wordTensors = await scentenceEncoderModel.embed(words);

    const t2 = performance.now();
    console.log(`Done! (in ${(t2 - t1).toFixed(0)} ms)`);

    return {
        wordTensors,
        words
    };
}

type TrainingData = {
    inputs: Tensor2D;
    expectedOutputs: Tensor2D;
};

// First, as an experiment
// Lets create a training dataset of n-grams and the expected next word
const buildTrainingData = async (): Promise<TrainingData> => {
    const text = readFileSync(CORPUS_PATH).toString();
    const words = wordpos.parse(text);
    const expectedOutputs = [];
    const inputs = [];

    for (let i = 0; i < words.length - BEFORE_SIZE - 1; i++) {
        const expectedOutput = words[i + BEFORE_SIZE];
        const ngrams = [];

        words.slice(i, i + BEFORE_SIZE).forEach(word => {
            ngrams.push(word);
        });

        if (ngrams.length < BEFORE_SIZE) {
            continue;
        }

        inputs.push(await words2Input(ngrams));

        // We want to predict the next word based on previous words.
        expectedOutputs.push(await embedWord(expectedOutput));
    }

    return {
        inputs: tf.concat(inputs),
        expectedOutputs: tf.concat(expectedOutputs)
    };
}

async function getTrainingData(): Promise<TrainingData> {
    console.log('building training data...');

    const PATH = 'data/trainingData.json';
    if (existsSync(PATH)) {
        return new Promise<TrainingData>(resolve => {
            const parseStream = json.createParseStream();
            parseStream.on('data', function(data) {
                data.inputs = tf.tensor(data.inputs);
                data.expectedOutputs = tf.tensor(data.expectedOutputs);
                resolve(data);
            });
            const readStream = createReadStream(PATH);
            readStream.pipe(parseStream);
        });
    }
    else {
        const data: TrainingData = await buildTrainingData();

        await new Promise<void>(resolve => {
            const body = {
                inputs: data.inputs.arraySync(),
                expectedOutputs: data.expectedOutputs.arraySync()
            };
            const stringifyStream = json.createStringifyStream({body});
            const writableStream = createWriteStream(PATH);

            stringifyStream.on('data', function(strChunk) {
                writableStream.write(strChunk);
            });
            stringifyStream.on('close', function() {
                resolve();
            });
        });

        return data;
    }
}

const wordCache = new Map<string, Tensor2D>();

async function embedWord(word: string): Promise<Tensor2D> {
    word = word.toLowerCase();
    if (!wordCache.has(word)) {
        const wordTensor = await scentenceEncoderModel.embed(word);
        wordCache.set(word, wordTensor);
    }

    return wordCache.get(word);
}

async function words2Input(ngrams) {
    const wordTensors = [];

    for (let i = 0; i < ngrams.length; i++) {
        wordTensors.push(await embedWord(ngrams[i]));
    }

    return tf.concat(wordTensors, 1);
}

function findClosestWord(tensor: Tensor2D, corpus: Corpus) {
    let closest = -1;
    let closestDist = null;

    for(let i = 0; i < corpus.wordTensors.shape[0]; i++) {
        const dist = tf.norm(tf.squaredDifference(corpus.wordTensors.slice(i, 1), tensor));
        if (closestDist === null || dist < closestDist) {
            closest = i;
            closestDist = dist;
        }
    }

    return corpus.words[closest];
}

async function buildModel() {
    const EPOCHS = 100;
    const ALPHA = 0.0015;

    const corpus = await buildCorpus();

    assert.equal(findClosestWord(await embedWord('horse'), corpus).indexOf('horse'), 0);

    console.log('Building embeddings');

    const wordPredictModel: Sequential = tf.sequential();
    const OUTPUT_WORD_COUNT = 1;
    const HIDDEN_SCALE = EMBED_SHAPE[1] * (BEFORE_SIZE + OUTPUT_WORD_COUNT);

    wordPredictModel.add(
        tf.layers.dense({
            inputShape: [EMBED_SHAPE[1] * 5],
            units: HIDDEN_SCALE,
            activation: "softmax",
        })
    );

    wordPredictModel.add(
        tf.layers.dropout({
            rate: 0.5
        })
    );

    wordPredictModel.add(
        tf.layers.dense({
            units: HIDDEN_SCALE,
            activation: "softmax",
        })
    );

    wordPredictModel.add(
        tf.layers.dropout({
            rate: 0.5
        })
    );

    wordPredictModel.add(
        tf.layers.dropout({
            rate: 0.5
        })
    );

    wordPredictModel.add(
        tf.layers.layerNormalization({})
    );


    wordPredictModel.add(
        tf.layers.dense({
            units: HIDDEN_SCALE,
            activation: "linear",
        })
    );

    wordPredictModel.add(
        tf.layers.dense({
            units: HIDDEN_SCALE,
            activation: "softmax",
        })
    );


    wordPredictModel.add(
        tf.layers.dense({
            units: EMBED_SHAPE[1],
            activation: "elu",
        })
    );

    wordPredictModel.summary();

    console.log('Compiling word prediction model.');
    wordPredictModel.compile({
        optimizer: tf.train.sgd(ALPHA),
        loss: "meanSquaredError",
    })
    console.log('Done!');

    console.log('Building training data!\n\n');

    const {
        inputs,
        expectedOutputs
    } = await getTrainingData();

    console.log('Built training data!');

    console.log('Training word prediction model.');

    await wordPredictModel.fit(inputs, expectedOutputs, {
        epochs: EPOCHS,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                if (epoch % 10 === 0) {
                    console.log(`Epoch ${epoch}: error: ${logs.loss}`)
                }
            },
        },
    });

    await wordPredictModel.save(WORD_PREDICT_MODEL_CACHE);
    return wordPredictModel;
}

async function getModel() {
    try {
        const wordPredictModel = await tf.loadLayersModel(WORD_PREDICT_MODEL_CACHE + '/model.json');
        return wordPredictModel;
    } catch (e) {
        console.log(e);
        console.log('Model not found/has error. Generating')
    }

    return buildModel();
}

export const main = async () => {
    const corpus = await buildCorpus();
    const wordPredictModel = await getModel() as LayersModel;

    // Test model
    const originalString = "The height of horses is measured";
    const words = wordpos.parse(originalString);
    for (let i = 0; i < 10; i++) {
        const last5 = words.slice(-BEFORE_SIZE);
        const wordTensors = await words2Input(last5);
        const prediction = wordPredictModel.predict(wordTensors) as Tensor2D;
        words.push(findClosestWord(prediction, corpus));
    }

    console.log(`Original string:  ${originalString}`);
    console.log(`Completed string: ${words.join(' ')}`);
};
