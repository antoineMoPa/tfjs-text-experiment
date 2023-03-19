import {
    readFileSync,
    existsSync,
    createWriteStream,
    createReadStream,
    readdirSync
} from 'fs';

import * as tf from '@tensorflow/tfjs-node';
import * as _ from 'lodash';
import { buildEncoderDecoder } from './encoderDecoder';
import { SymbolicTensor, Tensor } from '@tensorflow/tfjs-node';
import json from 'big-json';
import LRU from 'lru-cache';
import { FocusLayer } from './customLayers/focusLayer';

export const CORPUS_PATH = "data/corpus";

type WordIndexCache = LRU<string, tf.Tensor>;

export const tokenize = (text: string) => {
    const tokens = [];
    let cursor = 0;
    let lastTokenBeginCursor = 0;
    const stopRegex = /[ \n\\.]/;

    while (cursor < text.length) {
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

export type Vocabulary = {
    words: string[];
};

async function getVocabulary(): Promise<Vocabulary> {
    const PATH = 'data/vocabulary.json';
    if (existsSync(PATH)) {
        console.log('Reading vocabulary from cache...');

        return new Promise<Vocabulary>(resolve => {
            const parseStream = json.createParseStream();
            parseStream.on('data', function (data) {
                resolve(data);
            });
            const readStream = createReadStream(PATH);
            readStream.pipe(parseStream);
        });
    }
    else {
        const data: Vocabulary = buildVocabulary();
        console.log('Caching vocabulary to disk');
        await new Promise<void>(resolve => {
            const stringifyStream = json.createStringifyStream({ body: data });
            const writableStream = createWriteStream(PATH);

            stringifyStream.on('data', function (strChunk) {
                writableStream.write(strChunk);
            });
            stringifyStream.on('close', function () {
                resolve();
            });
        });
        return data;
    }
}

export function buildVocabulary(...texts: string[]): Vocabulary {
    let tokens = [];

    if (texts) {
        texts.forEach(text => {
            const newTokens = tokenize(text);
            tokens = [...tokens, ...newTokens];
        });
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
    return tf.oneHot(tf.tensor1d([index], 'int32'), vocabulary.words.length).cast('float32');
}

export function textToTensor(text: string, {
    vocabulary,
    encoderLayer,
    maxLength,
}: {
    vocabulary: Vocabulary
    encoderLayer: tf.layers.Layer,
    maxLength?: number
}) {
    const encodeWordIndexCache = new LRU<string, tf.Tensor2D>({
        max: 10,
        dispose(value: tf.Tensor) {
            value.dispose();
        }
    })

    let tokens = tokenize(text);

    if (tokens.length > maxLength) {
        tokens = tokens.slice(0, maxLength);
    }

    const tokenIndices = tokens.map(token => findWordIndex(token, vocabulary));
    const result = tf.concat(tokenIndices.map(
        index =>
            encodeWordIndex(
                index,
                {
                    vocabulary,
                    encoderLayer,
                    encodeWordIndexCache
                }
            ) as tf.Tensor
    ), 1);

    encodeWordIndexCache.clear();

    return result;
}

export function tokenIndicesToTensor(tokenIndices: number[], {
    vocabulary,
    encoderLayer,
}: {
    vocabulary: Vocabulary
    encoderLayer: tf.layers.Layer,
}) {
    const encodeWordIndexCache = new LRU<string, tf.Tensor2D>({
        max: 10,
        dispose(value: tf.Tensor) {
            value.dispose();
        }
    })

    const result = tf.concat(tokenIndices.map(
        index =>
            encodeWordIndex(
                index,
                {
                    vocabulary,
                    encoderLayer,
                    encodeWordIndexCache
                }
            ) as tf.Tensor
    ), 1);

    encodeWordIndexCache.clear();

    return result;
}


async function getTrainingData(
    { vocabulary, beforeSize }:
        { vocabulary: Vocabulary, beforeSize: number }
): Promise<TrainingData> {

    const PATH = 'data/trainingData.json';
    if (existsSync(PATH)) {
        console.log('Reading training data from cache...');

        return new Promise<TrainingData>(resolve => {
            const parseStream = json.createParseStream();
            parseStream.on('data', function (data) {
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
            stringifyStream.on('data', function (strChunk) {
                writableStream.write(strChunk);
            });
            stringifyStream.on('close', function () {
                resolve();
            });
        });
        console.log('Done!');
        return data;
    }
}

async function words2Input(ngrams, vocabulary) {
    const input = [];

    for (let i = 0; i < ngrams.length; i++) {
        const index = findWordIndex(ngrams[i], vocabulary);
        input.push(index);
    }

    return input;
}

export const buildTrainingData = async (
    { vocabulary, text, beforeSize }:
        { vocabulary: Vocabulary, text?: string, beforeSize: number }
): Promise<TrainingData> => {
    const expectedOutputs = [];
    const inputs = [];

    const buildTrainingDataForText = async (text) => {
        const words = tokenize(text);
        words.push('[END]');

        for (let i = 0; i < words.length - beforeSize; i += 1) {
            const inputWords = words.slice(i, i + beforeSize);
            const outputWords = words.slice(i + beforeSize, i + beforeSize + 1);

            inputs.push(await words2Input(inputWords, vocabulary));
            expectedOutputs.push(await words2Input(outputWords, vocabulary));

            if (i >= beforeSize) {
                inputWords.splice(0, 1);
                outputWords.splice(0, 1);
            }
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
    encodingSize?: number
    epochs?: number,
};

const encodeWordIndex = (
    index: number,
    {
        vocabulary,
        encoderLayer,
        encodeWordIndexCache
    }: {
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

const buildTimeStep = (
    encodingSize: number,
    beforeSize: number,
    offset: number
): tf.Tensor => {
    const data = [];

    // Using tf.linspace would not be cool, since the last item is the endpoint.
    // So we manually build the array.
    // numpy is better at this, since there is the `endpoint` option to np.linspace
    for (let j = 0; j < beforeSize; j++) {
        for (let i = 0; i < encodingSize; i++) {
            data.push(j + offset);
        }
    }

    return tf.tensor(data, [1, encodingSize * beforeSize], 'float32');
}

export function prepareModelInput({
    tokenIndices,
    vocabulary,
    encoderLayer,
    encodingSize,
    timestepOffset,
    mode,
    includeTimeStep,
}: {
    tokenIndices?: number[],
    vocabulary?: Vocabulary,
    encoderLayer?: tf.layers.Layer,
    encodingSize?: number,
    size?: number,
    timestepOffset: number,
    mode: 'onehot' | 'embedding',
    includeTimeStep: boolean
}) {
    if (mode === 'onehot') {
        const oneHot = tokenIndices.map(index => wordIndexToOneHot(index, vocabulary));
        const tensor = tf.concat(oneHot, 1);
        oneHot.map(t => t.dispose());

        if (includeTimeStep) {
            const timeSteps = buildTimeStep(vocabulary.words.length, tokenIndices.length, timestepOffset);
            const result = tf.concat([
                timeSteps,
                tensor,
            ], 0);

            timeSteps.dispose();
            tensor.dispose();

            return result;
        } else {
            return tensor;
        }
    } else {
        const tensor = tokenIndicesToTensor(tokenIndices, {
            vocabulary,
            encoderLayer,
        });

        if (includeTimeStep) {
            const timeSteps = buildTimeStep(encodingSize, tokenIndices.length, timestepOffset);

            const result = tf.concat([
                timeSteps,
                tensor,
            ], 0);

            tensor.dispose();
            timeSteps.dispose();

            return result;
        } else {
            return tensor;
        }
    }
}

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
        encodingSize = 128,
        epochs = 60,
    }: BuildModelArgs
): Promise<{
    wordPredictModel: tf.LayersModel;
    encoderLayer: tf.layers.Layer;
    decoderLayer: tf.layers.Layer;
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

    const inputs = tf.input({
        shape: [2, encodingSize * beforeSize],
    });

    let layerOutput: SymbolicTensor = inputs;

    const lstmTower = (layerOutputs, inputs = layerOutputs, unitsList = [128, 32, 128, 64, 128]) => {
        let towerOutput = layerOutputs;

        const stages = unitsList.map((units) => {
            const dense = () => {
                towerOutput = tf.layers.timeDistributed({
                    layer: tf.layers.dense({
                        units,
                        activation: 'relu',
                        kernelInitializer: tf.initializers.randomUniform({
                            minval: -0.01,
                            maxval: 0.01
                        }),
                        biasInitializer: tf.initializers.constant({ value: -0.01 }),
                    })
                }).apply(towerOutput) as SymbolicTensor;
            };

            dense();

            const lstmOutput = tf.layers.lstm({
                units,
                activation: 'relu',
                returnSequences: true,
                kernelInitializer: tf.initializers.randomUniform({
                    minval: -0.002,
                    maxval: 0.002
                }),
                recurrentInitializer: tf.initializers.randomUniform({
                    minval: -0.002,
                    maxval: 0.002
                }),
                biasInitializer: tf.initializers.constant({ value: -0.01 }),
                dropout: 0.05,
                recurrentDropout: 0,
            }).apply(towerOutput) as SymbolicTensor;

            towerOutput = tf.layers.concatenate().apply([
                inputs,
                lstmOutput,
            ]) as SymbolicTensor;

            return towerOutput;
        });

        return { towerOutput, stages };
    }

    // layerOutput = tf.layers.timeDistributed({
    //     layer:
    //         tf.layers.dense({
    //             units: 128,
    //             activation: 'relu',
    //             kernelInitializer: tf.initializers.randomUniform({ minval: -0.1, maxval: 0.1 }),
    //             biasInitializer: tf.initializers.constant({ value: -0.01 }),
    //         })
    // }).apply(layerOutput) as SymbolicTensor;

    const focusedLstmTower = ({ min, max  } : { min: number, max?: number}) => {
        return lstmTower(
            new FocusLayer({
                min,
                max,
                maxTimeStep: beforeSize,
            }).apply(layerOutput) as tf.SymbolicTensor,
            new FocusLayer({
                min,
                max,
                maxTimeStep: beforeSize,
            }).apply(inputs) as tf.SymbolicTensor
        ).towerOutput;
    };

    layerOutput = tf.layers.concatenate().apply([
        tf.layers.multiply().apply([
            focusedLstmTower({ min: 0, max: 0 }),
            focusedLstmTower({ min: 0 })
        ]) as tf.SymbolicTensor,
        focusedLstmTower({ min: 0, max: 2 }),
        focusedLstmTower({ min: 0, max: 8 }),
        focusedLstmTower({ min: 0, max: 16 }),
        focusedLstmTower({ min: 0, max: 0 }),
    ]) as tf.SymbolicTensor;

    layerOutput = tf.layers.timeDistributed({
        layer:
        tf.layers.dense({
            units: 128,
            activation: 'relu',
            kernelInitializer: tf.initializers.randomUniform({ minval: -0.2, maxval: 0.2 }),
            biasInitializer: tf.initializers.constant({ value: -0.01 }),
        })
    }).apply(layerOutput) as SymbolicTensor;

    const outputLayer =
        tf.layers.timeDistributed({
            layer:
                tf.layers.dense({
                    units: encodingSize,
                    activation: "relu",
                    kernelInitializer: tf.initializers.randomUniform({
                        minval: -0.08,
                        maxval: 0.08
                    }),
                    biasInitializer: tf.initializers.constant({ value: -0.01 }),
                    name: "output",
                })
        });

    layerOutput = outputLayer.apply(layerOutput) as SymbolicTensor;

    const outputs = layerOutput;
    const wordPredictModel = tf.model({ inputs, outputs });

    const alpha = 0.005;

    wordPredictModel.compile({
        optimizer: tf.train.adamax(alpha),
        loss: 'meanSquaredError',
    })

    const { encoderLayer, decoderLayer } = await buildEncoderDecoder({ vocabulary, encodingSize });

    const trainOnBatch = async (inputs, outputs) => {
        const trainingInputs = inputs.map(
            tokenIndices =>
                prepareModelInput({
                    tokenIndices,
                    vocabulary,
                    encoderLayer,
                    encodingSize,
                    timestepOffset: 0,
                    mode: 'embedding',
                    includeTimeStep: true
                })
        );
        const trainingOutputs = outputs.map(
            tokenIndices =>
                prepareModelInput({
                    tokenIndices,
                    vocabulary,
                    encoderLayer,
                    encodingSize,
                    timestepOffset: beforeSize,
                    mode: 'embedding',
                    includeTimeStep: true
                })
        );

        const concatenatedInput = tf.stack(trainingInputs);
        const concatenatedOutput = tf.stack(trainingOutputs);

        await wordPredictModel.fit(concatenatedInput, concatenatedOutput, {
            epochs,
            batchSize: 50,
            verbose: encodingSize > 35 ? 1 : 0,
            shuffle: true
        });

        [
            ...trainingInputs,
            ...trainingOutputs,
            concatenatedInput,
            concatenatedOutput
        ].forEach((tensor: tf.Tensor2D) => tensor.dispose());
    };

    const batchInputs = trainingData.inputs;
    const batchOutputs = trainingData.expectedOutputs;
    await trainOnBatch(batchInputs, batchOutputs);

    return {
        wordPredictModel,
        encoderLayer,
        decoderLayer,
        encodeWordIndexCache
    };
}

type BuildModelFromTextArgs = {
    text?: string,
    verbose?: boolean,
    level?: number,
    encodingSize?: number,
    epochs?: number
};

const LEVEL_TO_BEFORE_SIZE = {
    '0': 3,
    '1': 5,
    '2': 8,
};

export async function buildModelFromText({
    text,
    verbose,
    level,
    encodingSize,
    epochs
}: BuildModelFromTextArgs) {
    const beforeSize: number = LEVEL_TO_BEFORE_SIZE[level];
    const vocabulary = buildVocabulary(text);
    const trainingData = await buildTrainingData({
        vocabulary,
        text,
        beforeSize,
    });

    const {
        wordPredictModel,
        encoderLayer,
        encodeWordIndexCache,
        decoderLayer
    } = await buildModel({
        vocabulary,
        trainingData,
        verbose,
        level,
        beforeSize,
        encodingSize,
        epochs,
    });

    return {
        wordPredictModel,
        vocabulary,
        beforeSize,
        encoderLayer,
        decoderLayer,
        encodeWordIndexCache
    };
}

const oneHotWithTimestepDecoder = ({ stackedPredictionAndTimeStep, vocabulary }) => {
    const predictionAndTimeStep = tf.unstack(stackedPredictionAndTimeStep)[0];
    const prediction = predictionAndTimeStep.slice(
        [1, 0],
        [1, predictionAndTimeStep.shape[1]]
    );
    const token = tf.argMax(prediction, 1).dataSync()[0];
    const word = vocabulary.words[token];

    prediction.dispose();

    return { word, token };
}

const embeddingDecoder = ({ stackedPredictionAndTimeStep, vocabulary, decoderLayer }) => {
    const predictionAndTimeStep = tf.unstack(stackedPredictionAndTimeStep)[0];
    const embeddedPrediction = predictionAndTimeStep.slice(
        [1, 0],
        [1, predictionAndTimeStep.shape[1]]
    );
    const prediction = decoderLayer.apply(embeddedPrediction);
    const token = tf.argMax(prediction, 1).dataSync()[0];
    const word = vocabulary.words[token];

    prediction.dispose();

    return { word, token };
}

export const predict = async (before, {
    wordPredictModel,
    vocabulary,
    beforeSize,
    encoderLayer,
    decoderLayer,
    encodeWordIndexCache,
    encodingSize,
}: {
    wordPredictModel: tf.LayersModel,
    vocabulary: Vocabulary,
    beforeSize: number,
    encoderLayer: tf.layers.Layer,
    decoderLayer: tf.layers.Layer,
    encodeWordIndexCache: WordIndexCache
    encodingSize: number,
}) => {

    if (before.length < beforeSize) {
        console.error(`Before is not long enough. Got ${before.length}, expected ${beforeSize}. We'll pad the input with [NULL]`);

        while (before.length !== beforeSize) {
            before = ['[NULL]', ...before];
        }
    }

    const tokenIndices = await words2Input(before.slice(-beforeSize), vocabulary);

    const input = prepareModelInput({
        tokenIndices,
        vocabulary,
        encoderLayer,
        encodingSize,
        size: beforeSize,
        timestepOffset: 0,
        mode: 'embedding',
        includeTimeStep: true
    })

    const stackedPredictionAndTimeStep = wordPredictModel.predict(tf.stack([input])) as Tensor;
    const { word, token } = embeddingDecoder({ stackedPredictionAndTimeStep, vocabulary, decoderLayer })

    input.dispose();

    return { word, token };
}

export const predictUntilEnd = async (inputText, {
    vocabulary,
    wordPredictModel,
    beforeSize,
    encoderLayer,
    decoderLayer,
    encodeWordIndexCache,
    encodingSize
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
            decoderLayer,
            encodeWordIndexCache,
            encodingSize
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
    const {
        wordPredictModel,
        encoderLayer,
        encodeWordIndexCache,
        decoderLayer,
    } = await buildModel({
        vocabulary,
        trainingData,
        beforeSize,
        level: 0,
        verbose: true,
    });

    // Test model
    const originalString = "the quick brown";
    const result = await predictUntilEnd(originalString, {
        vocabulary,
        wordPredictModel,
        beforeSize,
        encoderLayer,
        decoderLayer,
        encodeWordIndexCache,
        encodingSize: 128
    })

    console.log(`Original string:  ${originalString}`);
    console.log(`Completed string: ${result}`);
};
