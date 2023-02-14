import * as tf from '@tensorflow/tfjs-node';
import { readFileSync, existsSync, createWriteStream, createReadStream, readdirSync } from 'fs';import type { Tensor2D } from '@tensorflow/tfjs-core/dist/tensor';

import {
    buildEncoderDecoder,
    buildVocabulary,
    CORPUS_PATH,
    wordIndexToOneHot,
} from '../src/tinygpt';

import { expect } from 'chai';
import { twoParagraphs } from './testText';

describe('Vocabulary EncoderDecoder', async () => {
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

    it('Encodes a tiny vocabulary', async function () {
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

    it('Encodes a large vocabulary', async function () {
        this.timeout(10000);

        // Arrange
        const text = twoParagraphs;
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

    it('Encodes entire text vocabulary', async function () {
        this.timeout(50000);

        // Arrange
        const text = readFileSync(CORPUS_PATH + '/wiki-horse.txt').toString();
        const vocabulary = await buildVocabulary(text);
        const encoderDecoder = await buildEncoderDecoder({ vocabulary });

        // Act
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
        expect(success).to.equal(vocabulary.words.length);
    });

    it('Encodes entire corpus vocabulary', async function () {
        this.timeout(50000);

        // Arrange
        const vocabulary = await buildVocabulary();
        const encoderDecoder = await buildEncoderDecoder({ vocabulary });

        // Act
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
        expect(success).to.equal(vocabulary.words.length);
    });
});
