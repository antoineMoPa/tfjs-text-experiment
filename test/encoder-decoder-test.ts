import * as tf from '@tensorflow/tfjs-node';
import { readFileSync } from 'fs';

import {
    buildVocabulary,
    CORPUS_PATH,
    wordIndexToOneHot,
} from '../src/tinygpt';

import { buildEncoderDecoder } from '../src/encoderDecoder';
import { expect } from 'chai';
import { twoParagraphs } from './testText';

describe('Vocabulary EncoderDecoder', async () => {
    it('Encodes a token', async function () {
        this.timeout(5000);

        // Arrange
        const text = 'the quick brown fox jumps over the lazy dog';
        const vocabulary = buildVocabulary(text);

        // Act
        const { encoderDecoder } = await buildEncoderDecoder({ vocabulary, encodingSize: 30 });
        const word0 = wordIndexToOneHot(0, vocabulary);
        const decoded = encoderDecoder.predict(word0) as tf.Tensor2D;
        const tokenIndex = tf.argMax(decoded, 1).dataSync()[0];

        // Assert
        expect(tokenIndex).to.equal(0);
    });

    it('Encodes a tiny vocabulary', async function () {
        this.timeout(5000);

        // Arrange
        const text = 'the quick brown fox jumps over the lazy dog';
        const vocabulary = buildVocabulary(text);
        const { encoderDecoder } = await buildEncoderDecoder({ vocabulary, encodingSize: 30 });

        // Act
        const encoded = [];
        const decoded = [];

        for (let i = 0; i < vocabulary.words.length; i++){
            const word = wordIndexToOneHot(i, vocabulary);
            const prediction = encoderDecoder.predict(word) as tf.Tensor2D;
            const tokenIndex = tf.argMax(prediction, 1).dataSync()[0] as number;

            encoded.push(i);
            decoded.push(tokenIndex);
        }

        // Assert
        expect(encoded).to.deep.equal(decoded);
    });

    it('Encodes a large vocabulary', async function () {
        this.timeout(10000);

        // Arrange
        const text = twoParagraphs;
        const vocabulary = buildVocabulary(text);
        const { encoderDecoder } = await buildEncoderDecoder({ vocabulary, encodingSize: 30 });

        // Act
        const encoded = [];
        const decoded = [];

        for (let i = 0; i < vocabulary.words.length; i++){
            const word = wordIndexToOneHot(i, vocabulary);
            const prediction = encoderDecoder.predict(word) as tf.Tensor2D;
            const tokenIndex = tf.argMax(prediction, 1).dataSync()[0] as number;

            encoded.push(i);
            decoded.push(tokenIndex);
        }

        // Assert
        expect(encoded).to.deep.equal(decoded);
    });

    it('Encodes entire text vocabulary', async function () {
        this.timeout(50000);

        // Arrange
        const text = readFileSync(CORPUS_PATH + '/wiki-horse.txt').toString();
        const vocabulary = buildVocabulary(text);
        const { encoderDecoder } = await buildEncoderDecoder({ vocabulary, encodingSize: 80 });

        // Act
        const encoded = [];
        const decoded = [];
        let success = 0;

        for (let i = 0; i < vocabulary.words.length; i++){
            const word = wordIndexToOneHot(i, vocabulary);
            const prediction = encoderDecoder.predict(word) as tf.Tensor2D;
            const tokenIndex = tf.argMax(prediction, 1).dataSync()[0] as number;

            encoded.push(i);
            decoded.push(tokenIndex);

            (i === tokenIndex) && success++;
        }

        // Assert
        expect(success).to.equal(vocabulary.words.length);
    });

    it('Encodes entire corpus vocabulary', async function () {
        this.timeout(50000);

        // Arrange
        const vocabulary = buildVocabulary();
        const { encoderDecoder } = await buildEncoderDecoder({ vocabulary, encodingSize: 128 });

        // Act
        const encoded = [];
        const decoded = [];
        let success = 0;

        for (let i = 0; i < vocabulary.words.length; i++){
            const word = wordIndexToOneHot(i, vocabulary);
            const prediction = encoderDecoder.predict(word) as tf.Tensor2D;
            const tokenIndex = tf.argMax(prediction, 1).dataSync()[0] as number;

            encoded.push(i);
            decoded.push(tokenIndex);

            (i === tokenIndex) && success++;
        }

        // Assert
        expect(success).to.equal(vocabulary.words.length);
    });
});
