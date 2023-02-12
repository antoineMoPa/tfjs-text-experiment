import { readFileSync, existsSync, createWriteStream, createReadStream, readdirSync } from 'fs';
import type { Tensor2D } from '@tensorflow/tfjs-core/dist/tensor';

import * as tf from '@tensorflow/tfjs-node';
import * as _ from 'lodash';
import { performance } from 'perf_hooks';
import { LayersModel, SymbolicTensor } from '@tensorflow/tfjs-node';
import * as json from 'big-json';

const CORPUS_PATH = "data/corpus";
const WORD_PREDICT_MODEL_CACHE = 'file://data/wordPredictModel';

export const tokenize = (text:string) => {
    const tokens = [];
    let cursor = 0;
    let lastTokenBeginCursor = 0;
    const stopRegex = /[ \n\\.]/;

    while(cursor < text.length) {
        if (stopRegex.test(text[cursor])) {
            const char = text[cursor];
            if (char === ' ' || char === '.' || char === '\n') {
                const token = text.slice(lastTokenBeginCursor, cursor);
                tokens.push(token);
                lastTokenBeginCursor = cursor;
            }
        }

        cursor += 1;
    }

    // Push any leftovers
    tokens.push(text.slice(lastTokenBeginCursor));

    return tokens;
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

export async function buildVocabulary(text?: string): Promise<Vocabulary> {
    console.log('Building vocabulary');
    const t1 = performance.now();
    let tokens = [];

    if (text) {
        tokens = tokenize(text);
    }
    else {
        const texts = readdirSync(CORPUS_PATH);

        for (const textFilename of texts) {
            const text = readFileSync(CORPUS_PATH + '/' + textFilename).toString();
            const newTokens = tokenize(text);
            tokens = [...tokens, ...newTokens];
        }
    }
    tokens.push('[END]');
    tokens.push('[NULL]');
    const words: string[] = _.shuffle(_.uniq(tokens));
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
    { vocabulary, beforeSize } :
    { vocabulary: Vocabulary, beforeSize: number }
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
        const data: TrainingData = await buildTrainingData({
            vocabulary,
            beforeSize
        });
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
    // console.log(`should be brown: ${(await predict(['the', 'quick'], wordPredictModel, vocabulary)).word}`);
    // console.log(`should be fox: ${(await predict(['quick', 'brown'], wordPredictModel, vocabulary)).word}`);
    // console.log(`should be over: ${(await predict(['fox', 'jumps'], wordPredictModel, vocabulary)).word}`);
    // console.log(`should be the: ${(await predict(['jumps', 'over'], wordPredictModel, vocabulary)).word}`);
};

async function words2Input(ngrams, vocabulary) {
    const input  = [];

    for (let i = 0; i < ngrams.length; i++) {
        const index = findWordIndex(ngrams[i], vocabulary);
        input.push(index);
    }

    return input;
}

export const buildTrainingData = async (
    { vocabulary, text, beforeSize } :
    { vocabulary: Vocabulary, text?: string, beforeSize: number }
): Promise<TrainingData> => {
    const expectedOutputs = [];
    const inputs = [];

    const buildTrainingDataForText = async (text) => {
        const words = tokenize(text);
        words.push('[END]');

        while (words.length > 1 && words.length > beforeSize) {
            const ngrams = [];
            let i = 0;
            for (; i < beforeSize && words.length > 1; i++){
                ngrams.push(words[i]);
            }

            const expectedOutput = words[beforeSize];
            inputs.push(await words2Input(ngrams, vocabulary));

            // We want to predict the next word based on previous words.
            const index = findWordIndex(expectedOutput, vocabulary);
            expectedOutputs.push(index);

            words.shift();
        }
    }

    if (text) {
        await buildTrainingDataForText(text);
    }
    else {
        const texts = readdirSync(CORPUS_PATH);

        for (const textFilename of texts) {
            const text = readFileSync(CORPUS_PATH + '/' + textFilename).toString();
            await buildTrainingDataForText(text);
        }
    }

    return {
        inputs,
        expectedOutputs,
    };
}

type BuildModelArgs = {
    vocabulary: Vocabulary,
    trainingData: TrainingData,
    verbose?: boolean,
    level?: number,
    /**
     * Amount of tokens that the model is able to read in input
     */
    beforeSize: number,
};

/**
 * buildModel
 *
 * Levels
 * With each level, the model grows in capacity. Smaller = lighter
 *
 * Levels:
 * 0: Able to remember 1 sentence
 * 1: Able to remember 2 long sentences
 *
 */
export async function buildModel(
    {
        vocabulary,
        trainingData,
        verbose,
        level,
        beforeSize
    } :BuildModelArgs
) {
    if (verbose === undefined) {
        verbose = true;
    }
    if (level === undefined) {
        level = 0;
    }

    const level_to_epochs = {
        '0': 100,
        '1': 100,
        '2': 100,
    };
    const level_to_alpha = {
        '0': 0.003,
        '1': 0.003,
        '2': 0.003
    };

    const epochs = level_to_epochs[level];

    const inputs = tf.input({
        shape: [(vocabulary.words.length) * beforeSize],
    });

    const denseLayer1 = tf.layers.dense({
        units: (vocabulary.words.length) * beforeSize,
        activation: "swish",
        kernelInitializer: tf.initializers.randomNormal({}),
        name: "denseLayer1",
    })

    const denseLayer1Output = denseLayer1.apply(inputs) as SymbolicTensor;
    let levelOutput = denseLayer1Output as SymbolicTensor;

    if (level > 0) {
        // const level1DenseLayer1 = tf.layers.dense({
        //     units: 3,
        //     activation: "swish",
        //     kernelInitializer: tf.initializers.randomNormal({}),
        //     name: "level1DenseLayer1",
        // });
        //
        // const level1DenseLayer1Output = level1DenseLayer1.apply(denseLayer1Output) as SymbolicTensor;
        //
        // levelOutput = level1DenseLayer1Output as SymbolicTensor;

        // // Feed back the last word in input
        // const concat = tf.layers.concatenate({ axis: 1 });
        // const combined = concat.apply([level1DenseLayer1Output, inputs]);
        //
        // levelOutput = combined as SymbolicTensor;
    }

    const outputLayer = tf.layers.dense({
        units: vocabulary.words.length,
        activation: "softmax",
        kernelInitializer: tf.initializers.randomNormal({}),
        name: "output",
    });

    const outputs = outputLayer.apply(levelOutput) as SymbolicTensor;
    const wordPredictModel =  tf.model({ inputs, outputs });

    verbose && wordPredictModel.summary();
    verbose && console.log('Compiling word prediction model.');

    const alpha = level_to_alpha[level];

    wordPredictModel.compile({
        optimizer: tf.train.adamax(alpha),
        loss: 'categoricalCrossentropy',
    })

    verbose && console.log('Building training data!\n\n');

    const trainingInputs = trainingData.inputs.map(
        sample => {
            const data = tf.concat(
                sample.map(value => wordIndexToOneHot(value, vocabulary)), 1
            );
            return data;
        }
    );
    const expectedOutputs = trainingData.expectedOutputs.map(
        value => wordIndexToOneHot(value, vocabulary)
    );

    verbose && console.log('Built training data!');
    verbose && console.log('Training word prediction model.');

    await wordPredictModel.fit(tf.concat(trainingInputs, 0), tf.concat(expectedOutputs, 0), {
        epochs,
        batchSize: 10,
        verbose: verbose ? 1 : 0,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                if (verbose && epoch % 10 === 0) {
                    console.log(`Epoch ${epoch}: error: ${logs.loss}`);
                    await minitest(wordPredictModel, vocabulary);
                }
            },
        },
    });

    await wordPredictModel.save(WORD_PREDICT_MODEL_CACHE);
    return wordPredictModel;
}

type BuildModelFromTextArgs = {
    text?: string,
    verbose?: boolean,
    level?: number,
};

const LEVEL_TO_BEFORE_SIZE = {
    '0': 3,
    '1': 5,
    '2': 10,
};

export async function buildModelFromText({
    text,
    verbose,
    level,
} : BuildModelFromTextArgs) {
    const beforeSize: number = LEVEL_TO_BEFORE_SIZE[level];
    const vocabulary = await buildVocabulary(text);
    const trainingData = await buildTrainingData({
        vocabulary,
        text,
        beforeSize,
    });

    const wordPredictModel = await buildModel({
        vocabulary,
        trainingData,
        verbose,
        level,
        beforeSize
    });

    return {
        wordPredictModel,
        vocabulary,
        beforeSize
    };
}

async function getModel(
    {
        vocabulary,
        trainingData,
        beforeSize
    } :
    {
        vocabulary: Vocabulary,
        trainingData: TrainingData,
        beforeSize: number
    }
) {
    try {
        const wordPredictModel = await tf.loadLayersModel(WORD_PREDICT_MODEL_CACHE + '/model.json');
        return wordPredictModel;
    } catch (e) {
        console.log('Model not found/has error. Generating')
    }

    return buildModel({ vocabulary, trainingData, beforeSize });
}

export const predict = async (before, {
    wordPredictModel,
    vocabulary,
    beforeSize
}) => {

    if (before.length < beforeSize) {
        console.error(`Before is not long enough. Got ${before.length}, expected ${beforeSize}. We'll pad the input with [NULL]`);

        while (before.length !== beforeSize) {
            before = ['[NULL]', ...before];
        }
    }

    const inputWords = await words2Input(before.slice(-beforeSize), vocabulary);
    const input = tf.concat(inputWords.map(value => wordIndexToOneHot(value, vocabulary)), 1);
    const prediction = wordPredictModel.predict(input) as Tensor2D;

    const token= tf.argMax(prediction, 1).dataSync()[0];

    const word = vocabulary.words[token];

    return { word, token };
}

export const predictUntilEnd = async (inputText, {
    vocabulary,
    wordPredictModel,
    beforeSize
}) => {
    // Test model
    const words = tokenize(inputText);
    const MAX = 200;
    let lastword = null
    for (let i = 0; i < MAX && lastword !== '[END]'; i++) {
        const { word } = await predict(words.slice(-beforeSize), { wordPredictModel, vocabulary, beforeSize});
        words.push(word);
        lastword = word;
    }

    return words.join('');
};


export const main = async () => {
    const vocabulary = await getVocabulary();
    const beforeSize = 3;
    const trainingData = await getTrainingData({
        vocabulary,
        beforeSize
    });
    const wordPredictModel = await getModel({
        vocabulary,
        trainingData,
        beforeSize
    }) as LayersModel;

    // Test model
    const originalString = "the quick brown";
    const result = await predictUntilEnd(originalString, {
        vocabulary,
        wordPredictModel,
        beforeSize
    })

    minitest(wordPredictModel, vocabulary)

    console.log(`Original string:  ${originalString}`);
    console.log(`Completed string: ${result}`);
};
