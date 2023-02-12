import { readFileSync, existsSync, createWriteStream, createReadStream } from 'fs';
import type { Tensor2D } from '@tensorflow/tfjs-core/dist/tensor';
import type { Sequential } from '@tensorflow/tfjs-layers/dist/models';

import * as tf from '@tensorflow/tfjs-node';
import * as _ from 'lodash';
import { performance } from 'perf_hooks';
import { LayersModel } from '@tensorflow/tfjs-node';
import * as json from 'big-json';

const BEFORE_SIZE = 2;
const CORPUS_PATH = "data/data-corpus.txt";
const WORD_PREDICT_MODEL_CACHE = 'file://data/wordPredictModel';

const tokenize = (text) => {
    return text.split(' ');
}

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
    const words: string[] = _.shuffle(_.uniq(tokenize(text)));
    words.push('[END]');
    words.push('[NULL]');
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

function wordIndexToOneHot(index: number, vocabulary: Vocabulary) {
    return tf.oneHot(tf.tensor1d([index], 'int32'), vocabulary.words.length);
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
    console.log(`should be brown: ${(await predict(['the', 'quick'], wordPredictModel, vocabulary)).word}`);
    console.log(`should be fox: ${(await predict(['quick', 'brown'], wordPredictModel, vocabulary)).word}`);
    console.log(`should be over: ${(await predict(['fox', 'jumps'], wordPredictModel, vocabulary)).word}`);
    console.log(`should be the: ${(await predict(['jumps', 'over'], wordPredictModel, vocabulary)).word}`);
};

async function words2Input(ngrams, vocabulary) {
    const input  = [];

    for (let i = 0; i < ngrams.length; i++) {
        const index = findWordIndex(ngrams[i], vocabulary);
        input.push(index);
    }

    return input;
}

const buildTrainingData = async (
    { vocabulary } :
    { vocabulary: Vocabulary }
): Promise<TrainingData> => {
    let text = readFileSync(CORPUS_PATH).toString();
    // HACK
    text = 'the quick brown fox jumps over the lazy dog';
    const words = tokenize(text);
    words.push('[END]');
    console.log({words})
    const expectedOutputs = [];
    const inputs = [];

    while (words.length > 1 && words.length > BEFORE_SIZE) {
        const ngrams = [];
        let i = 0;
        for (; i < BEFORE_SIZE && words.length > 1; i++){
            ngrams.push(words[i]);
        }

        const expectedOutput = words[BEFORE_SIZE];
        inputs.push(await words2Input(ngrams, vocabulary));

        // We want to predict the next word based on previous words.
        const index = findWordIndex(expectedOutput, vocabulary);
        expectedOutputs.push(index);

        console.log({ ngrams,  expectedOutput })

        words.shift();
    }

    return {
        inputs,
        expectedOutputs,
    };
}

async function buildModel(
    { vocabulary } :
    { vocabulary: Vocabulary }
) {
    const EPOCHS = 100;

    const wordPredictModel: Sequential = tf.sequential();

    wordPredictModel.add(
        tf.layers.dense({
            inputShape: [(vocabulary.words.length) * BEFORE_SIZE],
            units: (vocabulary.words.length) * BEFORE_SIZE,
            activation: "softmax",
            kernelInitializer: tf.initializers.randomNormal({}),
            name: "input",
        })
    );

    wordPredictModel.add(
        tf.layers.dense({
            units: vocabulary.words.length,
            activation: "softmax",
            kernelInitializer: tf.initializers.randomNormal({}),
            name: "output",
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
        sample => {
            const data = tf.concat(
                sample.map(value => wordIndexToOneHot(value, vocabulary)), 1
            );
            console.log(data.shape);
            return data;
        }
    );
    const expectedOutputs = data.expectedOutputs.map(
        value => wordIndexToOneHot(value, vocabulary)
    );

    // eslin
    debugger;

    console.log('Built training data!');

    console.log('Training word prediction model.');
    await wordPredictModel.fit(tf.concat(inputs, 0), tf.concat(expectedOutputs, 0), {
        epochs: EPOCHS,
        batchSize: 10,
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

    if (before.length !== BEFORE_SIZE) {
        throw new Error(`Before is not long enough. Got ${before.length}, expected ${BEFORE_SIZE}`);
    }

    const input = tf.concat(inputWords.map(value => wordIndexToOneHot(value, vocabulary)), 1);
    const prediction = wordPredictModel.predict(input) as Tensor2D;

    const token= tf.argMax(prediction, 1).dataSync()[0];

    const word = vocabulary.words[token];

    return { word, token };
}

export const main = async () => {
    const vocabulary = await getVocabulary();
    const wordPredictModel = await getModel({ vocabulary }) as LayersModel;

    // Test model
    const originalString = "the quick";
    const words = tokenize(originalString);

    for (let i = 0; i < 10; i++) {
        const { word } = await predict(words.slice(-BEFORE_SIZE), wordPredictModel, vocabulary);
        words.push(word);
        if (word === '[END]') {
            break;
        }
    }

    minitest(wordPredictModel, vocabulary)

    console.log(`Original string:  ${originalString}`);
    console.log(`Completed string: ${words.join(' ')}`);
};
