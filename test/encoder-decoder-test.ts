import * as tf from '@tensorflow/tfjs-node';
import type { Tensor2D } from '@tensorflow/tfjs-core/dist/tensor';

import {
    buildEncoderDecoder,
    buildVocabulary,
    wordIndexToOneHot
} from '../src/tinygpt';

import { expect } from 'chai';

describe.only('Vocabulary EncoderDecoder', async () => {
    it('Encodes a token', async function () {
        this.timeout(5000);

        // Arrange
        const text = 'the quick brown fox jumps over the lazy dog';
        const vocabulary = await buildVocabulary(text);

        // Act
        const encoderDecoder = await buildEncoderDecoder({ vocabulary });
        const word0 = wordIndexToOneHot(0, vocabulary);
        const decoded = encoderDecoder.predict(word0) as Tensor2D;
        const tokenIndex = tf.argMax(decoded, 1).dataSync()[0];

        // Assert
        expect(tokenIndex).to.equal(0);
    });

    it('Encodes all vocabulary', async function () {
        this.timeout(5000);

        // Arrange
        const text = 'the quick brown fox jumps over the lazy dog';
        const vocabulary = await buildVocabulary(text);
        const encoderDecoder = await buildEncoderDecoder({ vocabulary });

        // Act
        const encoded = [];
        const decoded = [];

        for (let i = 0; i < vocabulary.words.length; i++){
            const word = wordIndexToOneHot(i, vocabulary);
            const prediction = encoderDecoder.predict(word) as Tensor2D;
            const tokenIndex = tf.argMax(prediction, 1).dataSync()[0] as number;

            encoded.push(i);
            decoded.push(tokenIndex);
        }

        // Assert
        expect(encoded).to.deep.equal(decoded);
    });
});
