import {
    readFileSync,
    existsSync,
    createWriteStream,
    createReadStream,
    writeFileSync,
    readdirSync
} from 'fs';

import * as tf from '@tensorflow/tfjs-node';
import * as _ from 'lodash';
import { buildEncoderDecoder } from './encoderDecoder';
import { SymbolicTensor, Tensor } from '@tensorflow/tfjs-node';
import json from 'big-json';
import LRU from 'lru-cache';
import { focusDenseTower } from './layerCombos/focusDenseTower';

export const CORPUS_PATH = "data/corpus";

type EmbeddingType = 'onehot' | 'embedding';

const OUTPUT_FORMAT: EmbeddingType = 'onehot' as EmbeddingType;

type WordIndexCache = LRU<string, tf.Tensor>;

export const tokenize = (text: string) => {
    const tokens = [];
    let cursor = 0;
    let lastTokenBeginCursor = 0;
    const stopRegex = /[ \n\\.\\-]/;

    while (cursor < text.length) {
        if (stopRegex.test(text[cursor])) {
            const char = text[cursor];
            if (char === ' ' || char === '.' || char === '\n' || char === '-') {
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
    let words: string[] = _.shuffle(_.uniq(tokens));

    // Add null words to increase output space and allow new words in a next training
    words = [...words, ...Array(20).fill(null).map(() => '[nullword' + Math.random() + ']')];

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
        max: 3000,
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
    encodeWordIndexCache,
}: {
    vocabulary: Vocabulary;
    encoderLayer: tf.layers.Layer;
    encodeWordIndexCache: WordIndexCache;
}) {
    encodeWordIndexCache = encodeWordIndexCache || new LRU<string, tf.Tensor2D>({
        max: 1000,
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
    encodeWordIndexCache
}: {
    tokenIndices?: number[],
    vocabulary?: Vocabulary,
    encoderLayer?: tf.layers.Layer,
    encodingSize?: number,
    size?: number,
    timestepOffset: number,
    mode: 'onehot' | 'embedding',
    includeTimeStep: boolean;
    encodeWordIndexCache: WordIndexCache;
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
            encodeWordIndexCache
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

const minitest = async (inputText, {
    vocabulary,
    wordPredictModel,
    beforeSize,
    encoderLayer,
    decoderLayer,
    encodeWordIndexCache,
    encodingSize
}) => {
    return;
    const output = await predictUntilEnd(inputText, {
        vocabulary,
        wordPredictModel,
        beforeSize,
        encoderLayer,
        decoderLayer,
        encodeWordIndexCache,
        encodingSize
    });
    console.log(`minitest: ${output}`)
};

export const saveModel = async (name: string, {
    wordPredictModel,
    encoderDecoder,
    vocabulary,
    beforeSize,
    encodingSize,
} : {
    wordPredictModel: tf.LayersModel;
    encoderDecoder: tf.LayersModel;
    vocabulary: Vocabulary;
    beforeSize: number;
    encodingSize: number
}) => {
    await wordPredictModel.save(`file://${process.cwd()}/models/${name}_wordPredictModel`);
    await encoderDecoder.save(`file://${process.cwd()}/models/${name}_encoderDecoder`);

    const data = {
        vocabulary,
        beforeSize,
        encodingSize,
    };

    writeFileSync(`${process.cwd()}/models/${name}_data.json`, JSON.stringify(data), { encoding: 'utf8' });
};

export const loadModel = async (name: string) : Promise<{
    wordPredictModel: tf.LayersModel;
    encoderDecoder: tf.LayersModel;
    encoderLayer: tf.layers.Layer;
    decoderLayer: tf.layers.Layer;
    vocabulary: Vocabulary;
    beforeSize: number;
    encodingSize: number;
    encodeWordIndexCache: WordIndexCache;
}> => {
    const wordPredictModel = await tf.loadLayersModel(`file://${process.cwd()}/models/${name}_wordPredictModel/model.json`);
    const encoderDecoder = await tf.loadLayersModel(`file://${process.cwd()}/models/${name}_encoderDecoder/model.json`);

    const {
        vocabulary,
        beforeSize,
        encodingSize,
    } = JSON.parse(readFileSync(`${process.cwd()}/models/${name}_data.json`, { encoding: 'utf8' }));

    const encoderLayer = encoderDecoder.getLayer('encodedLayer');
    const decoderLayer = encoderDecoder.getLayer('decodedLayer');
    const encodeWordIndexCache = genCache();

    return {
        wordPredictModel,
        encoderDecoder,
        encoderLayer,
        decoderLayer,
        vocabulary,
        beforeSize,
        encodingSize,
        encodeWordIndexCache
    };
};

const genCache = () =>
    new LRU<string, tf.Tensor2D>({
        max: 5000,
        dispose(value: tf.Tensor) {
            value.dispose();
        }
    });

type BuildModelArgs = {
    vocabulary: Vocabulary;
    trainingData: TrainingData;
    verbose?: boolean;
    level?: number;
    /**
     * Amount of tokens that the model is able to read in input
     */
    beforeSize: number;
    encodingSize?: number
    epochs?: number;
    alpha?: number;
    minitestText?: string;
};

/**
 * buildModel
 */
export async function buildModel(
    {
        vocabulary,
        trainingData,
        verbose,
        level,
        beforeSize,
        encodingSize = 128,
        epochs = 25,
        alpha,
        minitestText,
    }: BuildModelArgs
): Promise<{
    wordPredictModel: tf.LayersModel;
    encoderLayer: tf.layers.Layer;
    decoderLayer: tf.layers.Layer;
    encodeWordIndexCache: WordIndexCache;
    encoderDecoder: tf.LayersModel,
}> {
    const encodeWordIndexCache = genCache();

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

    const unitsList = Array(4).fill(220);

    const towers = Array(beforeSize)
            .fill(1)
            .map(
                (_, index) =>
                    focusDenseTower({
                        min: 0,
                        max: index,
                        unitsList,
                        beforeSize,
                        layerOutput,
                        inputs,
                        vocabulary,
                    })
            )

    layerOutput = tf.layers.concatenate().apply(
        towers.map(t => t.towerOutput),
    ) as tf.SymbolicTensor;

    layerOutput = tf.layers.timeDistributed({
        layer: tf.layers.dense({
            units: 400,
            activation: 'relu',
            kernelInitializer: tf.initializers.randomUniform({
                minval: -0.5,
                maxval: 0.5
            }),
            trainable: false,
        })
    }).apply(layerOutput) as SymbolicTensor;

    let outputLayer = null;

    if (OUTPUT_FORMAT === 'embedding') {
        outputLayer =
            tf.layers.timeDistributed({
                layer:
                tf.layers.dense({
                    units: encodingSize,
                    activation: "relu",
                    kernelInitializer: tf.initializers.randomUniform({
                        minval: -0.08,
                        maxval: 0.08
                    }),
                    name: "output",
                })
            });
    } else {
        outputLayer =
            tf.layers.timeDistributed({
                layer:
                tf.layers.dense({
                    units: vocabulary.words.length,
                    activation: "softmax",
                    kernelInitializer: tf.initializers.randomUniform({
                        minval: -0.02,
                        maxval: 0.02
                    }),
                    name: "output",
                })
            });
    }

    layerOutput = outputLayer.apply(layerOutput) as SymbolicTensor;

    const outputs = layerOutput;
    const wordPredictModel = tf.model({ inputs, outputs });

    const { encoderDecoder, encoderLayer, decoderLayer } = await buildEncoderDecoder({ vocabulary, encodingSize });

    await trainModel({
        wordPredictModel,
        vocabulary,
        trainingData,
        verbose,
        beforeSize,
        encodingSize,
        epochs,
        minitestText,
        encoderLayer,
        decoderLayer,
        encodeWordIndexCache,
        alpha
    });

    return {
        wordPredictModel,
        encoderLayer,
        decoderLayer,
        encodeWordIndexCache,
        encoderDecoder
    };
}

type TrainModelArgs = {
    wordPredictModel: tf.LayersModel;
    vocabulary: Vocabulary;
    trainingData: TrainingData;
    verbose: boolean;
    beforeSize: number;
    encodingSize: number;
    epochs: number;
    minitestText?: string;
    encodeWordIndexCache: WordIndexCache;
    encoderLayer: tf.layers.Layer;
    decoderLayer: tf.layers.Layer;
    alpha?: number;
};

type TrainModelWithTextArgs = Omit<TrainModelArgs, 'trainingData'>
    & { text: string };

/**
 * trainModelWithText
 * Note that this can mutate vocabulary and wordPredictModel
 */
export const trainModelWithText = async ({
    text,
    vocabulary,
    wordPredictModel,
    verbose,
    beforeSize,
    encodingSize,
    epochs,
    minitestText,
    encoderLayer,
    decoderLayer,
    encodeWordIndexCache,
    alpha,
}: TrainModelWithTextArgs) => {
    alpha = alpha || 0.005;
    const tokens = tokenize(text);

    for (const t of tokens) {
        if (vocabulary.words.indexOf(t) === -1) {
            const index = vocabulary.words.findIndex(
                token =>
                    token.startsWith('[nullword')
            );

            if (index > 0) {
                console.log(`adding ${t} at position ${index}`);
                vocabulary.words[index] = t;
                continue;
            }

            throw new Error(`Cannot train: word '${t}' is not in original vocabulary and no space is left in output space.`);
        }
    }

    const trainingData = await buildTrainingData({
        vocabulary,
        text,
        beforeSize,
    });

    await trainModel({
        wordPredictModel,
        vocabulary,
        trainingData,
        verbose,
        beforeSize,
        encodingSize,
        epochs,
        minitestText,
        encoderLayer,
        decoderLayer,
        encodeWordIndexCache,
        alpha
    });
}

export const trainModel = async ({
    wordPredictModel,
    vocabulary,
    trainingData,
    verbose,
    beforeSize,
    encodingSize,
    epochs,
    minitestText,
    encoderLayer,
    decoderLayer,
    encodeWordIndexCache,
    alpha = 0.002,
}: TrainModelArgs) => {
    wordPredictModel.compile({
        optimizer: tf.train.adamax(alpha, 0.01, 0.9999),
        loss: 'categoricalCrossentropy',
    });

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
                    includeTimeStep: true,
                    encodeWordIndexCache,
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
                    mode: OUTPUT_FORMAT,
                    includeTimeStep: true,
                    encodeWordIndexCache,
                })
        );

        const concatenatedInput = tf.stack(trainingInputs);
        const concatenatedOutput = tf.stack(trainingOutputs);

        await wordPredictModel.fit(concatenatedInput, concatenatedOutput, {
            epochs,
            batchSize: 30,
            verbose: encodingSize >= 30 ? 1 : 0,
            shuffle: true,
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    if (minitestText && verbose && epoch % 10 === 8) {
                        await minitest(minitestText, {
                            vocabulary,
                            wordPredictModel,
                            beforeSize,
                            encoderLayer,
                            decoderLayer,
                            encodeWordIndexCache,
                            encodingSize
                        });
                    }
                },
            },
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
}

type BuildModelFromTextArgs = {
    text?: string;
    verbose?: boolean;
    level?: number;
    encodingSize?: number;
    epochs?: number;
    beforeSize?: number;
    alpha?: number
};

export async function buildModelFromText({
    text,
    verbose,
    level,
    encodingSize,
    epochs,
    beforeSize,
    alpha
}: BuildModelFromTextArgs) {
    const LEVEL_TO_BEFORE_SIZE = {
        '0': 3,
        '1': 30,
    };
    beforeSize = beforeSize || LEVEL_TO_BEFORE_SIZE[level];
    const vocabulary = buildVocabulary(text);
    const trainingData = await buildTrainingData({
        vocabulary,
        text,
        beforeSize,
    });
    const {
        wordPredictModel,
        encoderDecoder,
        encoderLayer,
        decoderLayer,
        encodeWordIndexCache,
    } = await buildModel({
        vocabulary,
        trainingData,
        verbose,
        level,
        beforeSize,
        encodingSize,
        epochs,
        alpha,
        minitestText: tokenize(text).slice(0, beforeSize).join(' ')
    });

    return {
        wordPredictModel,
        vocabulary,
        beforeSize,
        encoderDecoder,
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
        includeTimeStep: true,
        encodeWordIndexCache
    })

    const stackedPredictionAndTimeStep = wordPredictModel.predict(tf.stack([input])) as Tensor;

    input.dispose();

    if (OUTPUT_FORMAT === 'embedding') {
        return embeddingDecoder({ stackedPredictionAndTimeStep, vocabulary, decoderLayer })
    } else {
        return oneHotWithTimestepDecoder({ stackedPredictionAndTimeStep, vocabulary })
    }
}

export const predictUntilEnd = async (inputText, {
    vocabulary,
    wordPredictModel,
    beforeSize,
    encoderLayer,
    decoderLayer,
    encodeWordIndexCache,
    encodingSize,
    continousLogging
}: {
    vocabulary: Vocabulary
    wordPredictModel: tf.LayersModel;
    beforeSize: number;
    encoderLayer: tf.layers.Layer;
    decoderLayer: tf.layers.Layer;
    encodeWordIndexCache: WordIndexCache;
    encodingSize: number;
    continousLogging?: boolean;
}) => {
    // Test model
    const words = tokenize(inputText);
    const MAX = 3000;
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

        continousLogging && process.stdout.write('\n\n\n\n' + words.join('') + '\n');
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
    const originalString = "the quick brown";

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
        minitestText: originalString
    });

    // Test model
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
