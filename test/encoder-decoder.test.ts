import * as tf from '@tensorflow/tfjs-node';
import { readFileSync } from 'fs';

import {
    buildVocabulary,
    CORPUS_PATH,
    wordIndexToOneHot,
} from '../src/model';

import { buildEncoderDecoder } from '../src/encoderDecoder';
import { describe, it, expect } from 'vitest';
import { twoParagraphs } from './testText';

describe('Vocabulary EncoderDecoder', async () => {
    it('Encodes a token', async function () {
        // Arrange
        const text = 'the quick brown fox jumps over the lazy dog';
        const vocabulary = buildVocabulary(text);

        // Act
        const { encoderDecoder } = await buildEncoderDecoder({
            vocabulary,
            encodingSize: 30,
        });
        const word0 = wordIndexToOneHot(0, vocabulary);
        const decoded = encoderDecoder.predict(word0) as tf.Tensor2D;
        const tokenIndex = tf.argMax(decoded, 1).dataSync()[0];

        // Assert
        expect(tokenIndex).to.equal(0);
    }, 5000);

    it('Encodes a tiny vocabulary', async function () {
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
    }, 5000);

    it('Encodes a large vocabulary', async function () {
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
    }, 10000);

    it('Encodes entire text vocabulary', async function () {
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
    }, 50000);

    it('Encodes entire corpus vocabulary', async function () {
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
    }, 50000);
});
