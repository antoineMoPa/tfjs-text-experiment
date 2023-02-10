import { readFileSync, existsSync, createWriteStream, createReadStream } from 'fs';
import type { Tensor2D } from '@tensorflow/tfjs-core/dist/tensor';
import type { Sequential } from '@tensorflow/tfjs-layers/dist/models';
import type { UniversalSentenceEncoder } from  '@tensorflow-models/universal-sentence-encoder';
import * as tf from '@tensorflow/tfjs-node';
import * as _ from 'lodash';
import { performance } from 'perf_hooks';
import { LayersModel } from '@tensorflow/tfjs-node';
import * as json from 'big-json';
import * as assert from 'node:assert';
import * as WordPOS from 'wordpos';

const EMBED_SHAPE = [1, 512];
const BEFORE_SIZE = 128;
const CORPUS_PATH = "data/data-corpus.txt";
const scentenceEncoderModel: UniversalSentenceEncoder = global.universalSentenceEncoderModel;
const WORD_PREDICT_MODEL_CACHE = 'file://data/wordPredictModel';
const wordpos = new WordPOS({ stopwords: false });

type Vocabulary = {
    wordTensors: Tensor2D;
    words: string[];
};

// Should randomize the order of words
async function buildVocabulary(): Promise<Vocabulary> {
    console.log('Building vocabulary');
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

function findWordIndex(expectedOutput: string, vocabulary: Vocabulary) {
    return vocabulary.words.findIndex(item => item === expectedOutput);
}

type TrainingData = {
    inputs: Tensor2D;
    expectedOutputs: Tensor2D;
};

// First, as an experiment
// Lets create a training dataset of n-grams and the expected next word
const buildTrainingData = async (
    { vocabulary } :
    { vocabulary: Vocabulary }
): Promise<TrainingData> => {
    const text = readFileSync(CORPUS_PATH).toString();
    const words = wordpos.parse(text);
    const expectedOutputs = [];
    const inputs = [];

    for (let i = 0; i < words.length - BEFORE_SIZE - 1; i++) {
        if (i % 500 === 0) {
            console.log(`built ${(i/words.length*100).toFixed(0)}% of training data`);
        }
        if (Math.random() > 0.01) {
            continue;
        }
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
        const index = findWordIndex(expectedOutput, vocabulary);
        console.log(index);
        expectedOutputs.push(tf.oneHot(tf.tensor1d([index], 'int32'), vocabulary.words.length));
    }

    console.log(tf.concat(inputs).shape, tf.concat(expectedOutputs).shape);

    return {
        inputs: tf.concat(inputs),
        expectedOutputs: tf.concat(expectedOutputs)
    };
}

async function getTrainingData(
    { vocabulary } :
    { vocabulary: Vocabulary }
): Promise<TrainingData> {

    const PATH = 'data/trainingData.json';
    if (existsSync(PATH)) {
        console.log('Reading training data from cache...');

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
        const data: TrainingData = await buildTrainingData({ vocabulary });
        console.log('Caching training data to disk');
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
        console.log('Done!');
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

    // Normalize input to always have BEFORE_SIZE samples
    while (wordTensors.length < BEFORE_SIZE) {
        wordTensors.push(tf.zeros(EMBED_SHAPE));
    }

    return tf.concat(wordTensors, 1);
}

function findClosestWord(tensor: Tensor2D, vocabulary: Vocabulary) {
    let closest = -1;
    let closestDist = null;

    for(let i = 0; i < vocabulary.wordTensors.shape[0]; i++) {
        const dist = tf.norm(tf.squaredDifference(vocabulary.wordTensors.slice(i, 1), tensor));
        if (closestDist === null || dist < closestDist) {
            closest = i;
            closestDist = dist;
        }
    }

    return vocabulary.words[closest];
}

async function buildModel(
    { vocabulary } :
    { vocabulary: Vocabulary }
) {
    const EPOCHS = 2;

    assert.equal(findClosestWord(await embedWord('horse'), vocabulary).indexOf('horse'), 0);

    console.log('Building embeddings');

    const wordPredictModel: Sequential = tf.sequential();

    const HIDDEN_SCALE = 512;

    wordPredictModel.add(
        tf.layers.dense({
            inputShape: [EMBED_SHAPE[1] * BEFORE_SIZE],
            units: HIDDEN_SCALE * 10,
            activation: "softmax",
            kernelInitializer: tf.initializers.zeros()
        })
    );

    wordPredictModel.add(
        tf.layers.dropout({
            rate: 0.5
        })
    );

    wordPredictModel.add(
        tf.layers.dense({
            units: HIDDEN_SCALE * 2,
            activation: "softmax",
            kernelInitializer: tf.initializers.zeros()
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
            activation: "softmax",
            kernelInitializer: tf.initializers.zeros()
        })
    );

    wordPredictModel.add(
        tf.layers.dense({
            units: vocabulary.words.length,
            activation: "elu",
        })
    );

    wordPredictModel.summary();

    console.log('Compiling word prediction model.');
    wordPredictModel.compile({
        optimizer: tf.train.adam(),
        loss: 'binaryCrossentropy',
    })

    console.log('Done!');

    console.log('Building training data!\n\n');

    const {
        inputs,
        expectedOutputs
    } = await getTrainingData({ vocabulary });
    console.log(expectedOutputs.shape)


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

async function getModel(
    { vocabulary } :
    { vocabulary: Vocabulary }
) {
    try {
        const wordPredictModel = await tf.loadLayersModel(WORD_PREDICT_MODEL_CACHE + '/model.json');
        return wordPredictModel;
    } catch (e) {
        console.log(e);
        console.log('Model not found/has error. Generating')
    }

    return buildModel({ vocabulary });
}

export const main = async () => {
    const vocabulary = await buildVocabulary();
    const wordPredictModel = await getModel({ vocabulary }) as LayersModel;

    // Test model
    const originalString = "The height of horses is measured";
    const words = wordpos.parse(originalString);
    for (let i = 0; i < 10; i++) {
        const last5 = words.slice(-BEFORE_SIZE);
        const wordTensors = await words2Input(last5);
        const prediction = wordPredictModel.predict(wordTensors) as Tensor2D;

        const predictedWord = tf.argMax(prediction, 1).dataSync()[0];
        console.log(predictedWord);
        words.push(vocabulary.words[predictedWord]);
    }

    console.log(`Original string:  ${originalString}`);
    console.log(`Completed string: ${words.join(' ')}`);
};
