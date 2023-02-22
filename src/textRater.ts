import type { Tensor2D } from '@tensorflow/tfjs-core/dist/tensor';
import * as tf from '@tensorflow/tfjs-node';
import { SymbolicTensor } from '@tensorflow/tfjs-node';
import { buildVocabulary } from '../src/tinygpt';

import { tokenize } from './tinygpt';
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
} : {
    text: string,
}) {
    const words = tokenize(text)
        .slice(0, TEXT_RATER_INPUT_LENGTH)
        .map(word => word.trim());
    let { words: vocabulary } = buildVocabulary(text);
    vocabulary = vocabulary.map(word => word.trim());
    const encodingSize = TEXT_RATER_INPUT_LENGTH;

    const tensor = tf.concat(
        words.map(
            word => {
                return tf.oneHot(
                    tf.tensor1d([vocabulary.indexOf(word)], 'int32'), encodingSize, 1, 0, 'float32'
                )
            }
        ),
        1
    );

    return tensor;
}

export function rateText(
    text: string,
    {
        textRater,
    } :
    {
        textRater: tf.LayersModel;
    }
): TEXT_RATER_OUTPUT {
    const tensor = prepareTextRaterInput({
        text,
    });
    const output = textRater.predict(tensor) as tf.Tensor;
    const result = tf.argMax(output, 1).dataSync()[0];
    return result;
}

export async function buildTextRater() {
    const inputLength = TEXT_RATER_INPUT_LENGTH;
    const encodingSize = TEXT_RATER_INPUT_LENGTH;
    const inputs = tf.input({
        shape: [inputLength * encodingSize],
        name: 'input'
    });

    const outputSize = TEXT_RATER_OUTPUT_LEN;

    let layerOutput = inputs;

    const rnnLayer = tf.layers.dense({
        units: 20,
        activation: 'swish',
        kernelInitializer: tf.initializers.randomNormal({}),
    });

    layerOutput = rnnLayer.apply(inputs) as SymbolicTensor;

    const rnnLayer2 = tf.layers.dense({
        units: 16,
        activation: 'swish',
        kernelInitializer: tf.initializers.randomNormal({}),
    });

    layerOutput = rnnLayer2.apply(layerOutput) as SymbolicTensor;

    const outputLayer = tf.layers.dense({
        units: outputSize,
        activation: "softmax",
        name: "output",
        kernelInitializer: tf.initializers.zeros(),
    });

    const outputs = outputLayer.apply(layerOutput) as SymbolicTensor;

    const textRater = tf.model({ inputs, outputs });

    const alpha = 0.001;
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
                })
            );

            // Each category (like REPETITIVE) is represented by one unit.
            expectedOutputs.push(
                tf.oneHot(tf.tensor1d([i], 'int32'), TEXT_RATER_OUTPUT_LEN, 1, 0, 'float32')
            );
        });
    }

    const concatenatedInput = tf.concat(trainingInputs, 0);
    const concatenatedOutput = tf.concat(expectedOutputs, 0);

    const epochs = 10;

    await textRater.fit(concatenatedInput, concatenatedOutput, {
        epochs,
        batchSize: 10,
        shuffle: true,
        verbose: 1,
    });

    [
        ...trainingInputs,
        ...expectedOutputs,
        concatenatedInput,
        concatenatedOutput
    ].forEach((tensor: Tensor2D) => tensor.dispose());

    return { textRater };
}
