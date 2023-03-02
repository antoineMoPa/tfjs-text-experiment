import type { Tensor2D } from '@tensorflow/tfjs-core/dist/tensor';
import * as tf from '@tensorflow/tfjs-node';
import { SymbolicTensor } from '@tensorflow/tfjs-node';
import { Vocabulary, wordIndexToOneHot } from './tinygpt';

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

    const decoderLayer = tf.layers.dense({
        units: vocabulary.words.length,
        activation: "softmax",
        kernelInitializer: tf.initializers.randomNormal({}),
        name: "output",
    });

    const outputs = decoderLayer.apply(encodedLayerOutput) as SymbolicTensor;
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

    return {
        encoderDecoder,
        encoderLayer,
        decoderLayer
    };
}
