import type { Tensor2D } from '@tensorflow/tfjs-core/dist/tensor';
import * as tf from '@tensorflow/tfjs-node';
import { SymbolicTensor } from '@tensorflow/tfjs-node';
import {
    Vocabulary,
    textToTensor,
    tokenize
} from './tinygpt';
export const TEXT_RATER_INPUT_LENGTH = 8;

// No output means that the text is good
export enum TEXT_RATER_OUTPUT {
    NEVER = 0,
    REPETITIVE = 1,
    GOOD = 2,
}

export const TEXT_RATER_OUTPUT_VALUES = [
    TEXT_RATER_OUTPUT.NEVER,
    TEXT_RATER_OUTPUT.REPETITIVE,
    TEXT_RATER_OUTPUT.GOOD
];

import { textRaterData } from './textRaterData';

// Null is not part of the output
export const TEXT_RATER_OUTPUT_LEN = TEXT_RATER_OUTPUT_VALUES.length;

export function prepareTextRaterInput({
    text,
    vocabulary,
    encoderLayer,
    encodingSize
} : {
    text: string,
    vocabulary: Vocabulary,
    encoderLayer: tf.layers.Layer,
    encodingSize: number
}) {
    const inputLength = TEXT_RATER_INPUT_LENGTH;
    const inputSize = encodingSize * inputLength;
    const timeSteps = tf
        .linspace(0, inputLength, inputSize)
        .reshape([1, inputSize])
        .floor();

    const tensor = textToTensor(text, {
        vocabulary,
        encoderLayer,
        maxLength: inputLength
    });

    const result = tf.concat([
        timeSteps,
        tensor,
    ], 0);

    return result;
}

export function rateText(
    text: string,
    {
        vocabulary,
        encoderLayer,
        textRater,
        encodingSize
    } :
    {
        vocabulary: Vocabulary;
        encoderLayer: tf.layers.Layer;
        textRater: tf.LayersModel;
        encodingSize: number
    }
): TEXT_RATER_OUTPUT {
    const tensor = prepareTextRaterInput({
        text,
        vocabulary,
        encoderLayer,
        encodingSize
    });
    const output = textRater.predict(tensor.reshape([
        1,
        tensor.shape[0],
        tensor.shape[1],
    ])) as tf.Tensor;
    const result = tf.argMax(output, 1).dataSync()[0];
    return result;
}

export async function buildTextRater(
    {
        vocabulary,
        encoderLayer,
        encodingSize,
    }: {
        vocabulary: Vocabulary;
        encoderLayer: tf.layers.Layer;
        encodingSize: number
    }
) {
    const inputLength = TEXT_RATER_INPUT_LENGTH;
    const inputs = tf.input({
        shape: [2, inputLength * encodingSize],
        name: 'input'
    });

    const outputSize = TEXT_RATER_OUTPUT_LEN;

    let layerOutput = inputs;

    const rnnLayer = tf.layers.lstm({
        units: 100,
        activation: 'sigmoid',
        kernelInitializer: tf.initializers.randomUniform({}),
    });

    layerOutput = rnnLayer.apply(layerOutput) as SymbolicTensor;

    const outputLayer = tf.layers.dense({
        units: outputSize,
        activation: "softmax",
        name: "output",
        kernelInitializer: tf.initializers.randomUniform({}),
    });

    layerOutput = outputLayer.apply(layerOutput) as SymbolicTensor;

    const outputs = layerOutput;

    const textRater = tf.model({ inputs, outputs });

    const alpha = 0.008;
    textRater.compile({
        optimizer: tf.train.adamax(alpha),
        loss: 'categoricalCrossentropy', // categoricalCrossentropy meanSquaredError
    })

    const trainingInputs = [];
    const expectedOutputs = [];

    for (let i = 0; i < Object.keys(textRaterData).length; i++) {
        textRaterData[i].forEach(text => {
            const tokens = tokenize(text).slice(0, inputLength);
            if (tokens.length !== inputLength) {
                throw new Error(`Expected ${inputLength} tokens!`);
            }
            trainingInputs.push(
                prepareTextRaterInput({
                    text,
                    vocabulary,
                    encoderLayer,
                    encodingSize
                })
            );

            // Each category (like REPETITIVE) is represented by one unit.
            expectedOutputs.push(
                tf.oneHot(tf.tensor1d([i], 'int32'), TEXT_RATER_OUTPUT_LEN)
            );
        });
    }

    const concatenatedInput = tf.stack(trainingInputs);
    const concatenatedOutput = tf.concat(expectedOutputs, 0);

    console.log(concatenatedInput.shape)

    const params = [
        {epochs: 20, batchSize: 20},
    ];

    for (const param of params) {
        const {epochs, batchSize} = param;
        await textRater.fit(concatenatedInput, concatenatedOutput, {
            epochs,
            batchSize,
            shuffle: true,
            verbose: 1,
        });
    }

    [
        ...trainingInputs,
        ...expectedOutputs,
        concatenatedInput,
        concatenatedOutput
    ].forEach((tensor: Tensor2D) => tensor.dispose());

    return { textRater };
}
