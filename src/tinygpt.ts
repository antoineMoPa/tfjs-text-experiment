import { readFileSync, existsSync, createWriteStream, createReadStream } from 'fs';
import type { Tensor2D } from '@tensorflow/tfjs-core/dist/tensor';
import type { Sequential } from '@tensorflow/tfjs-layers/dist/models';

import * as tf from '@tensorflow/tfjs-node';
import * as _ from 'lodash';
import { performance } from 'perf_hooks';
import { LayersModel } from '@tensorflow/tfjs-node';
import * as json from 'big-json';
import * as WordPOS from 'wordpos';

const BEFORE_SIZE = 1;
const CORPUS_PATH = "data/data-corpus.txt";
const WORD_PREDICT_MODEL_CACHE = 'file://data/wordPredictModel';
const wordpos = new WordPOS({ stopwords: false });

type Vocabulary = {
    words: string[];
};

async function getVocabulary(): Promise<Vocabulary> {
    const PATH = 'data/vocabulary.json';
    if (existsSync(PATH)) {
        console.log('Reading vocabulary from cache...');

        return new Promise<Vocabulary>(resolve => {
            const parseStream = json.createParseStream();
            parseStream.on('data', function(data) {
                resolve(data);
            });
            const readStream = createReadStream(PATH);
            readStream.pipe(parseStream);
        });
    }
    else {
        const data: Vocabulary =  await buildVocabulary();
        console.log('Caching vocabulary to disk');
        await new Promise<void>(resolve => {
            const stringifyStream = json.createStringifyStream({ body: data });
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


// Should randomize the order of words
async function buildVocabulary(): Promise<Vocabulary> {
    console.log('Building vocabulary');
    const t1 = performance.now();
    let text = readFileSync(CORPUS_PATH).toString();
    // hack
    text = 'the quick brown fox jumps over the lazy dog';
    const words: string[] = _.shuffle(_.uniq(wordpos.parse(text)));

    const t2 = performance.now();
    console.log(`Done! (in ${(t2 - t1).toFixed(0)} ms)`);

    return { words };
}

function findWordIndex(expectedOutput: string, vocabulary: Vocabulary) {
    return vocabulary.words.findIndex(item => item === expectedOutput);
}

type TrainingData = {
    inputs: number[][];
    expectedOutputs: number[];
};

function indexToOneHot(index: number, vocabulary: Vocabulary) {
    return tf.oneHot(tf.tensor1d([index], 'int32'), vocabulary.words.length);
}

// First, as an experiment
// Lets create a training dataset of n-grams and the expected next word
const buildTrainingData = async (
    { vocabulary } :
    { vocabulary: Vocabulary }
): Promise<TrainingData> => {
    let text = readFileSync(CORPUS_PATH).toString();
    // HACK
    text = 'the quick brown fox jumps over the lazy dog';

    const words = wordpos.parse(text);
    const expectedOutputs = [];
    const inputs = [];

    for (let i = 0; i < words.length - BEFORE_SIZE - 1; i++) {
        if (i % 500 === 0) {
            console.log(`built ${(i/words.length*100).toFixed(0)}% of training data`);
        }
        const expectedOutput = words[i + BEFORE_SIZE];
        const ngrams = [];

        words.slice(i, i + BEFORE_SIZE).forEach(word => {
            ngrams.push(word);
        });

        if (ngrams.length < BEFORE_SIZE) {
            continue;
        }

        inputs.push(await words2Input(ngrams, vocabulary));

        // We want to predict the next word based on previous words.
        const index = findWordIndex(expectedOutput, vocabulary);
        expectedOutputs.push(index);
    }

    return {
        inputs,
        expectedOutputs
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
            const stringifyStream = json.createStringifyStream({ body: data });
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

const minitest = async (wordPredictModel, vocabulary) => {
    console.log(`should be quick: ${await predict(['the'], wordPredictModel, vocabulary)}`);
    console.log(`should be fox: ${await predict(['brown'], wordPredictModel, vocabulary)}`);
    console.log(`should be over: ${await predict(['jumps'], wordPredictModel, vocabulary)}`);
};

async function words2Input(ngrams, vocabulary) {
    let input  = [];

    for (let i = 0; i < ngrams.length; i++) {
        const index = findWordIndex(ngrams[i], vocabulary);
        input.push(index);
    }

    // Normalize input to always have BEFORE_SIZE samples
    while (input.length < BEFORE_SIZE) {
        input = [0, ...input];
    }

    return input;
}

async function buildModel(
    { vocabulary } :
    { vocabulary: Vocabulary }
) {
    const EPOCHS = 100;

    const wordPredictModel: Sequential = tf.sequential();

    const HIDDEN_SCALE = 20;

    wordPredictModel.add(
        tf.layers.dense({
            inputShape: [vocabulary.words.length * BEFORE_SIZE],
            units: HIDDEN_SCALE,
            activation: "softmax",
            kernelInitializer: tf.initializers.randomNormal({})
        })
    );

    wordPredictModel.add(
        tf.layers.dense({
            units: vocabulary.words.length,
            activation: "softmax",
            kernelInitializer: tf.initializers.randomNormal({})
        })
    );

    wordPredictModel.summary();

    console.log('Compiling word prediction model.');
    const alpha = 0.003;
    wordPredictModel.compile({
        optimizer: tf.train.adamax(alpha),
        loss: 'categoricalCrossentropy',
    })

    console.log('Building training data!\n\n');

    const data = await getTrainingData({ vocabulary });

    const inputs = data.inputs.map(
        sample =>
            tf.concat(
                sample.map(value => indexToOneHot(value, vocabulary)),
                1
            )
    );
    const expectedOutputs = data.expectedOutputs.map(
        value => indexToOneHot(value, vocabulary)
    );

    console.log('Built training data!');

    console.log('Training word prediction model.');
    console.log({ inputs });
    await wordPredictModel.fit(tf.concat(inputs,0), tf.concat(expectedOutputs,0), {
        epochs: EPOCHS,
        batchSize: 1,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                await minitest(wordPredictModel, vocabulary);

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
        console.log('Model not found/has error. Generating')
    }

    return buildModel({ vocabulary });
}

const predict = async (before, wordPredictModel, vocabulary: Vocabulary) => {
    const inputWords = await words2Input(before.slice(-BEFORE_SIZE), vocabulary);

    const input = tf.concat(inputWords.map(value => indexToOneHot(value, vocabulary)));
    const prediction = wordPredictModel.predict(input) as Tensor2D;

    const predictedWord = tf.argMax(prediction, 1).dataSync()[0];

    return vocabulary.words[predictedWord]
}

export const main = async () => {
    const vocabulary = await getVocabulary();
    const wordPredictModel = await getModel({ vocabulary }) as LayersModel;

    // Test model
    const originalString = "the";
    const words = wordpos.parse(originalString);

    for (let i = 0; i < 10; i++) {
        words.push(await predict(words, wordPredictModel, vocabulary));
    }

    minitest(wordPredictModel, vocabulary)

    console.log(`Original string:  ${originalString}`);
    console.log(`Completed string: ${words.join(' ')}`);
};
