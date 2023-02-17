import type { Tensor2D } from '@tensorflow/tfjs-core/dist/tensor';
import * as tf from '@tensorflow/tfjs-node';
import { SymbolicTensor } from '@tensorflow/tfjs-node';
import {
    Vocabulary,
    textToTensor,
    tokenize
} from './tinygpt';
import { textRaterData } from './textRaterData';
import { tokenProbabilities } from './metrics/entropy';

// No output means that the text is good
export const enum TEXT_RATER_OUTPUT {
    REPETITIVE = 0,
    GOOD = 1,
}
// Null is not part of the output
const TEXT_RATER_OUTPUT_LEN = 1;

export function rateText(
    text: string,
    {
        vocabulary,
        encoderLayer,
        textRater,
    } :
    {
        vocabulary: Vocabulary;
        encoderLayer: tf.layers.Layer;
        textRater: tf.LayersModel;
    }
): TEXT_RATER_OUTPUT {
    const tensor = textToTensor(text, {
        vocabulary,
        encoderLayer,
        maxLength: 10
    });
    const output = textRater.predict(tensor) as tf.Tensor;
    console.log(output.dataSync()[0]);

    const result = tf.argMax(output, 1).dataSync()[0];
    const probability = output.dataSync()[result];
    const threshold = 0;

    if (probability > threshold) {
        return result;
    }

    return TEXT_RATER_OUTPUT.GOOD;
}

export async function buildTextRater(
    {
        vocabulary,
        inputSize,
        encodingSize,
        encoderLayer,
    }: {
        vocabulary: Vocabulary;
        inputSize: number;
        encodingSize?: number;
        encoderLayer: tf.layers.Layer;
    }
) {
    const encodedInputSize = encodingSize * inputSize;
    const inputs = tf.input({
        shape: [encodedInputSize],
    });

    const denseLayer1 = tf.layers.dense({
        units: encodedInputSize,
        activation: "swish",
        kernelInitializer: tf.initializers.randomNormal({}),
        name: "denseLayer1",
    })

    const denseLayer1Output = denseLayer1.apply(inputs) as SymbolicTensor;

    const denseLayer2 = tf.layers.dense({
        units: 30,
        activation: "swish",
        name: "denseLayer2",
        kernelInitializer: tf.initializers.randomNormal({}),
    });

    const denseLayer2Output = denseLayer2.apply(denseLayer1Output) as SymbolicTensor;

    const denseLayer3 = tf.layers.dense({
        units: 30,
        activation: "swish",
        name: "denseLayer3",
        kernelInitializer: tf.initializers.randomNormal({}),
    });

    const denseLayer3Output = denseLayer3.apply(denseLayer2Output) as SymbolicTensor;

    // const normLayer2 = tf.layers.layerNormalization();
    // const normLayer2Output = normLayer2.apply(denseLayer2Output);

    const outputLayer = tf.layers.dense({
        units: TEXT_RATER_OUTPUT_LEN,
        activation: "swish",
        name: "output",
        kernelInitializer: tf.initializers.randomNormal({}),
    });

    // const normLayer3 = tf.layers.layerNormalization();
    // const normLayer3Output = normLayer3.apply(outputLayer.apply(normLayer2Output));
    // const outputs = normLayer3Output as SymbolicTensor;
    const outputs = outputLayer.apply(denseLayer3Output) as SymbolicTensor;

    const textRater = tf.model({ inputs, outputs });

    const alpha = 0.01;
    textRater.compile({
        optimizer: tf.train.adamax(alpha),
        loss: 'meanSquaredError',
    })

    const trainingInputs = [];
    const expectedOutputs = [];

    for (let i = 0; i < Object.keys(textRaterData).length; i++) {
        textRaterData[i].forEach(text => {
            const tokens = tokenize(text).slice(0, 10);
            if (tokens.length !== 10) {
                throw new Error('Expected 10 tokens!');
            }
            trainingInputs.push(
                textToTensor(
                    tokens.join(''), { vocabulary, encoderLayer }
                )
            );

            // Each category (like REPETITIVE) is reprensented by one unit.
            // GOOD is represented by having zeros as output
            if (i === TEXT_RATER_OUTPUT.GOOD) {
                console.log('IS GOODDDD');
                expectedOutputs.push(tf.zeros([TEXT_RATER_OUTPUT_LEN]));
            } else if (TEXT_RATER_OUTPUT_LEN === 1) {
                console.log('IS BADD');
                expectedOutputs.push(tf.mul(tf.ones([TEXT_RATER_OUTPUT_LEN]), 1));
            } else {
                console.log('IS NONE OF THE ABOVE');
                expectedOutputs.push(
                    tf.oneHot(tf.tensor1d([i], 'int32'), TEXT_RATER_OUTPUT_LEN)
                );
            }
        });
    }
    const concatenatedInput = tf.concat(trainingInputs, 0);
    const concatenatedOutput = tf.concat(expectedOutputs, 0);

    const epochs = 30;

    await textRater.fit(concatenatedInput, concatenatedOutput, {
        epochs,
        batchSize: 1,
        shuffle: true,
        verbose: 1
    });

    [
        ...trainingInputs,
        ...expectedOutputs,
        concatenatedInput,
        concatenatedOutput
    ].forEach((tensor: Tensor2D) => tensor.dispose());

    return { textRater };
}
