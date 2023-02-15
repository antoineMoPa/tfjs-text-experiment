import { readFileSync, existsSync, createWriteStream, createReadStream, readdirSync } from 'fs';
import type { Tensor2D } from '@tensorflow/tfjs-core/dist/tensor';

import * as tf from '@tensorflow/tfjs-node';
import * as _ from 'lodash';
import { performance } from 'perf_hooks';
import { LayersModel, SymbolicTensor } from '@tensorflow/tfjs-node';
import * as json from 'big-json';
import * as LRU from 'lru-cache';

export const CORPUS_PATH = "data/corpus";
const WORD_PREDICT_MODEL_CACHE = 'file://data/wordPredictModel';

type WordIndexCache = LRU<string, tf.Tensor>;

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
        return data;
    }
}

export async function buildVocabulary(text?: string): Promise<Vocabulary> {
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

export function wordIndexToOneHot(index: number, vocabulary: Vocabulary) {
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

export async function buildEncoderDecoder(
    {
        vocabulary,
        encodingSize = 128
    }: {
        vocabulary: Vocabulary;
        encodingSize?: number;
    }
) {
    const bigVocab = vocabulary.words.length > 100;
    const inputs = tf.input({
        shape: [(vocabulary.words.length)],
    });

    const encoderLayer = tf.layers.dense({
        units: encodingSize,
        activation: "swish",
        kernelInitializer: tf.initializers.randomNormal({}),
        name: "encodedLayer",
    })

    const encodedLayerOutput = encoderLayer.apply(inputs) as SymbolicTensor;

    const outputLayer = tf.layers.dense({
        units: vocabulary.words.length,
        activation: "softmax",
        kernelInitializer: tf.initializers.randomNormal({}),
        name: "output",
    });

    const outputs = outputLayer.apply(encodedLayerOutput) as SymbolicTensor;
    const encoderDecoder =  tf.model({ inputs, outputs });

    encoderDecoder.compile({
        optimizer: tf.train.adamax(0.05),
        loss: 'categoricalCrossentropy',
    })

    const trainingInputs = [];
    const expectedOutputs = [];

    for (let i = 0; i < vocabulary.words.length; i++) {
        trainingInputs.push(wordIndexToOneHot(i, vocabulary));
        expectedOutputs.push(wordIndexToOneHot(i, vocabulary));
    }

    const concatenatedInput = tf.concat(trainingInputs, 0);
    const concatenatedOutput = tf.concat(expectedOutputs, 0);
    const epochs = vocabulary.words.length < 100 ? 100: 8;

    await encoderDecoder.fit(concatenatedInput, concatenatedOutput, {
        epochs,
        batchSize: 1500,
        shuffle: true,
        verbose: bigVocab ? 1 : 0,
    });

    [
        ...trainingInputs,
        ...expectedOutputs,
        concatenatedInput,
        concatenatedOutput
    ].forEach((tensor: Tensor2D) => tensor.dispose());

    // Measure encoding/decoding success rate
    const encoded = [];
    const decoded = [];
    let success = 0;

    for (let i = 0; i < vocabulary.words.length; i++){
        const word = wordIndexToOneHot(i, vocabulary);
        const prediction = encoderDecoder.predict(word) as Tensor2D;
        const tokenIndex = tf.argMax(prediction, 1).dataSync()[0] as number;

        encoded.push(i);
        decoded.push(tokenIndex);

        (i === tokenIndex) && success++;
    }

    // Assert
    const total = vocabulary.words.length;
    console.log(`Encoder/Decoder success rate: (${(success/total * 100).toFixed(0)}%)`);
    if (success !== total) {
        throw new Error('Encoder/Decoder was not successful at encoding vocabulary.');
    }

    await encoderDecoder.save(WORD_PREDICT_MODEL_CACHE);

    return { encoderDecoder, encoderLayer };
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
    encodingSize?: number
};

const encodeWordIndex = (
    index: number,
    {
        vocabulary,
        encoderLayer,
        encodeWordIndexCache
    } : {
        vocabulary: Vocabulary;
        encoderLayer: tf.layers.Layer;
        encodeWordIndexCache: WordIndexCache;
    }
) => {
    const key = `${index}-${vocabulary.words.length}`;
    if (encodeWordIndexCache.has(key)) {
        return encodeWordIndexCache.get(key);
    }
    else {
        const oneHot = wordIndexToOneHot(index, vocabulary);
        const result = encoderLayer.apply(oneHot);
        oneHot.dispose();

        encodeWordIndexCache.set(key, result as tf.Tensor);
        return result;
    }
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
        beforeSize,
        encodingSize = 128
    } : BuildModelArgs
): Promise<{
    wordPredictModel: tf.LayersModel;
    encoderLayer: tf.layers.Layer;
    encodeWordIndexCache: WordIndexCache;
}> {
    const encodeWordIndexCache = new LRU<string, tf.Tensor2D>({
        max: 1000,
        dispose(value: tf.Tensor) {
            value.dispose();
        }
    })

    if (verbose === undefined) {
        verbose = true;
    }
    if (level === undefined) {
        level = 1;
    }

    // We can delete levels if the next level is more efficient and more compact.
    const level_to_epochs = {
        '1': 4,
        '2': 10,
    };
    const level_to_meta_epochs = {
        '1': 5,
        '2': 10,
    };
    const level_to_batch_size = {
        '1': 10,
        '2': 10,
    };
    const level_to_meta_batch_size = {
        '1': 10,
        '2': 10,
    };
    const level_to_alpha = {
        '1': 0.0015,
        '2': 0.003
    };
    const level_to_denseLayer1_size = {
        '1': 1500,
        '2': 8000
    };

    const epochs = level_to_epochs[level];
    const meta_epochs = level_to_meta_epochs[level];
    const inputs = tf.input({
        shape: [encodingSize * beforeSize],
    });

    const denseLayer1 = tf.layers.dense({
        units: level_to_denseLayer1_size[level],
        activation: "swish",
        kernelInitializer: tf.initializers.randomNormal({}),
        name: "denseLayer1",
    })

    const denseLayer1Output = denseLayer1.apply(inputs) as SymbolicTensor;
    let levelOutput = denseLayer1Output as SymbolicTensor;

    if (level > 1) {
        const level1DenseLayer1 = tf.layers.dense({
            units: 6000,
            activation: "swish",
            kernelInitializer: tf.initializers.randomNormal({}),
            name: "level1DenseLayer1",
        });

        const level1DenseLayer1Output = level1DenseLayer1.apply(denseLayer1Output) as SymbolicTensor;

        levelOutput = level1DenseLayer1Output as SymbolicTensor;
    }

    const outputLayer = tf.layers.dense({
        units: vocabulary.words.length,
        activation: "softmax",
        kernelInitializer: tf.initializers.randomNormal({}),
        name: "output",
    });

    const outputs = outputLayer.apply(levelOutput) as SymbolicTensor;
    const wordPredictModel =  tf.model({ inputs, outputs });

    const alpha = level_to_alpha[level];

    wordPredictModel.compile({
        optimizer: tf.train.adamax(alpha),
        loss: 'categoricalCrossentropy',
    })

    const { encoderLayer } = await buildEncoderDecoder({ vocabulary, encodingSize });


    const trainOnBatch = async (inputs, outputs) => {
        const trainingInputs = inputs.map(
            sample => {
                const data = tf.concat(
                    sample.map(value => encodeWordIndex(value, { vocabulary, encoderLayer, encodeWordIndexCache })),
                    1
                );
                return data;
            }
        );
        const expectedOutputs = outputs.map(
            value => wordIndexToOneHot(value, vocabulary)
        );

        const concatenatedInput = tf.concat(trainingInputs, 0);
        const concatenatedOutput = tf.concat(expectedOutputs, 0);
        const batchSize = level_to_batch_size[level];

        await wordPredictModel.fit(concatenatedInput, concatenatedOutput, {
            epochs,
            batchSize,
            verbose: 0,
        });

        [
            ...trainingInputs,
            ...expectedOutputs,
            concatenatedInput,
            concatenatedOutput
        ].forEach((tensor: Tensor2D) => tensor.dispose());
    };

    // To save memory and process all corpus,
    // we fit the model on parts of the corpus and repeat the process.
    // So in addition to batching and epochs in wordPredictModel.fit,
    // We have batches and epochs at this higher level,
    // which is where the "meta" comes from.
    const metaBatchSize = level_to_meta_batch_size[level];

    // Training data expands to a lot of memory. Split training so we don't have a lot
    // at the time.
    for (let i = 0; i < meta_epochs; i++) {
        verbose && console.log(`Meta epoch ${i}/${meta_epochs}.`);
        for (let j = 0; j < trainingData.inputs.length; j += metaBatchSize) {
            await trainOnBatch(trainingData.inputs.slice(j, j + metaBatchSize),
                               trainingData.expectedOutputs.slice(j, j + metaBatchSize));
        }
    }

    await wordPredictModel.save(WORD_PREDICT_MODEL_CACHE);

    return { wordPredictModel, encoderLayer, encodeWordIndexCache };
}

type BuildModelFromTextArgs = {
    text?: string,
    verbose?: boolean,
    level?: number,
    encodingSize?: number
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
    encodingSize
} : BuildModelFromTextArgs) {
    const beforeSize: number = LEVEL_TO_BEFORE_SIZE[level];
    const vocabulary = await buildVocabulary(text);
    const trainingData = await buildTrainingData({
        vocabulary,
        text,
        beforeSize,
    });

    const { wordPredictModel, encoderLayer, encodeWordIndexCache } = await buildModel({
        vocabulary,
        trainingData,
        verbose,
        level,
        beforeSize,
        encodingSize
    });

    return {
        wordPredictModel,
        vocabulary,
        beforeSize,
        encoderLayer,
        encodeWordIndexCache
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
    beforeSize,
    encoderLayer,
    encodeWordIndexCache
}: {
    wordPredictModel: tf.LayersModel,
    vocabulary: Vocabulary,
    beforeSize: number,
    encoderLayer: tf.layers.Layer,
    encodeWordIndexCache: WordIndexCache
}) => {

    if (before.length < beforeSize) {
        console.error(`Before is not long enough. Got ${before.length}, expected ${beforeSize}. We'll pad the input with [NULL]`);

        while (before.length !== beforeSize) {
            before = ['[NULL]', ...before];
        }
    }

    const inputWords = await words2Input(before.slice(-beforeSize), vocabulary);

    const input = tf.concat(inputWords.map(
        value =>
            encodeWordIndex(
                value,
                {
                    vocabulary,
                    encoderLayer,
                    encodeWordIndexCache: encodeWordIndexCache
                }
            ) as tf.Tensor
    ), 1);

    const prediction = wordPredictModel.predict(input) as Tensor2D;

    const token= tf.argMax(prediction, 1).dataSync()[0];

    const word = vocabulary.words[token];

    return { word, token };
}

export const predictUntilEnd = async (inputText, {
    vocabulary,
    wordPredictModel,
    beforeSize,
    encoderLayer,
    encodeWordIndexCache
}) => {
    // Test model
    const words = tokenize(inputText);
    const MAX = 200;
    let lastword = null
    for (let i = 0; i < MAX && lastword !== '[END]'; i++) {
        const { word } = await predict(words.slice(-beforeSize), {
            wordPredictModel,
            vocabulary,
            beforeSize,
            encoderLayer,
            encodeWordIndexCache
        });
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
    const { wordPredictModel, encoderLayer, encodeWordIndexCache } = await getModel({
        vocabulary,
        trainingData,
        beforeSize,
    }) as any;

    // Test model
    const originalString = "the quick brown";
    const result = await predictUntilEnd(originalString, {
        vocabulary,
        wordPredictModel,
        beforeSize,
        encoderLayer,
        encodeWordIndexCache
    })

    minitest(wordPredictModel, vocabulary)

    console.log(`Original string:  ${originalString}`);
    console.log(`Completed string: ${result}`);
};
