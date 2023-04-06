import { describe, it, expect} from 'vitest';

import { readFileSync } from 'fs';

import {
    tokenize,
    buildVocabulary,
    buildTrainingData,
    buildModel,
    predict,
    predictUntilEnd,
    buildModelFromText,
    CORPUS_PATH,
    serializeModel,
    loadModel,
    trainModelWithText
} from '../src/model';

import {
    _1Paragraph,
    _2Paragraphs,
    _3Paragraphs,
    _4Paragraphs,
    _8Paragraphs,
    _16Paragraphs,
    otherParagraph
} from './testText';

describe('Model', async () => {
    it.skip('Should build a vocabulary', async () => {
        // Arrange
        const text = 'the quick brown fox jumps over the lazy dog';

        // Act
        const vocabulary = buildVocabulary(text);

        // Assert
        expect(vocabulary.words).to.contain(' fox');
        expect(vocabulary.words).to.contain('[END]');
    });

    it('Should remember a simple word', async function () {
        // Arrange
        const text = 'the quick brown fox jumps over the lazy dog';
        const vocabulary = buildVocabulary(text);
        const beforeSize = 3;
        const trainingData = await buildTrainingData({ vocabulary, text, beforeSize });

        const {
            wordPredictModel,
            encoderLayer,
            decoderLayer,
            encodeWordIndexCache
        } = await buildModel({
            vocabulary,
            trainingData,
            verbose: false,
            beforeSize,
            encodingSize: 7,
            epochs: 25,
        });

        // Act
        const { word } = await predict(
            tokenize("the quick brown"),
            {
                wordPredictModel,
                vocabulary,
                beforeSize,
                encoderLayer, decoderLayer,
                encodeWordIndexCache,
                encodingSize: 7,
            }
        );

        // Assert
        expect(word, 'The quick brown [?]').to.equal(' fox');
    }, 10000);

    it('Should save a model', async function () {
        // Arrange
        const text = 'the quick brown fox jumps over the lazy dog';
        const vocabulary = buildVocabulary(text);
        const beforeSize = 2;
        const trainingData = await buildTrainingData({ vocabulary, text, beforeSize });
        const encodingSize = 7;

        const {
            wordPredictModel,
            encoderDecoder
        } = await buildModel({
            vocabulary,
            trainingData,
            verbose: false,
            beforeSize,
            encodingSize,
            epochs: 30,
        });

        await serializeModel('theQuickBrownFox', {
            wordPredictModel,
            encoderDecoder,
            vocabulary,
            beforeSize,
            encodingSize,
        });
    }, 10000);

    it('Should load a model', async function () {
        // Arrange
        const {
            wordPredictModel,
            encoderLayer,
            decoderLayer,
            vocabulary,
            beforeSize,
            encodingSize,
            encodeWordIndexCache,
        } = await loadModel('theQuickBrownFox');

        // Act
        const { word } = await predict(
            tokenize("the quick brown"),
            {
                wordPredictModel,
                vocabulary,
                beforeSize,
                encoderLayer,
                decoderLayer,
                encodeWordIndexCache,
                encodingSize,
            }
        );

        // Assert
        expect(word, 'The quick brown [?]').to.equal(' fox');
    }, 10000);

    it('Should fit a model on new data', async function () {
        // Arrange
        const {
            wordPredictModel,
            encoderLayer,
            decoderLayer,
            vocabulary,
            beforeSize,
            encodingSize,
            encodeWordIndexCache,
        } = await loadModel('theQuickBrownFox');

        // Act
        await trainModelWithText({
            text: 'the quick brown dog jumps over the lazy fox',
            vocabulary,
            wordPredictModel,
            verbose: true,
            beforeSize,
            encodingSize,
            encodeWordIndexCache,
            encoderLayer,
            decoderLayer,
            epochs: 20,
            alpha: 0.005,
        });

        const { word } = await predict(
            tokenize("the quick brown"),
            {
                wordPredictModel,
                vocabulary,
                beforeSize,
                encoderLayer,
                decoderLayer,
                encodeWordIndexCache,
                encodingSize,
            }
        );

        // Assert
        expect(word, 'The lazy brown [?]').to.equal(' dog');
    }, 10000);


    it('Should remember a simple sentence', async function () {
        // Arrange
        const text = 'the quick brown fox jumps over the lazy dog';
        const vocabulary = buildVocabulary(text);
        const beforeSize = 3;
        const trainingData = await buildTrainingData({ vocabulary, text, beforeSize });
        const { wordPredictModel, encoderLayer, decoderLayer,  encodeWordIndexCache } = await buildModel({
            vocabulary,
            trainingData,
            verbose: false,
            beforeSize,
            encodingSize: 10
        });

        // Act
        const sentence = await predictUntilEnd("the quick brown", {
            vocabulary,
            wordPredictModel,
            beforeSize,
            encoderLayer, decoderLayer,
            encodeWordIndexCache,
            encodingSize: 10
        })

        // Assert
        expect(sentence).to.equal(text + '[END]');
    }, 10000);

    it('Should remember a more complex sentence', async function () {
        // Arrange
        const text = 'It belongs to the taxonomic family Equidae and is one of two extant subspecies of Equus ferus.';
        const vocabulary = buildVocabulary(text);
        const beforeSize = 3;
        const trainingData = await buildTrainingData({ vocabulary, text, beforeSize,  });
        const {
            wordPredictModel,
            encoderLayer,
            decoderLayer,
            encodeWordIndexCache
        } = await buildModel({
            vocabulary,
            trainingData,
            verbose: false,
            beforeSize,
            encodingSize: 10,
        });

        // Act
        const sentence = await predictUntilEnd("It belongs to", {
            vocabulary,
            wordPredictModel,
            beforeSize,
            encoderLayer, decoderLayer,
            encodeWordIndexCache,
            encodingSize: 10
        })

        // Assert
        expect(sentence).to.equal(text + '[END]');
    }, 10000);

    it('Should remember an even more complex sentence', async function () {
        // Arrange
        const text = 'The horse has evolved over the past 45 to 55 million years from a small multi-toed creature, Eohippus, into the large, single-toed animal of today';
        const {
            wordPredictModel,
            vocabulary,
            beforeSize,
            encoderLayer,
            decoderLayer,
            encodeWordIndexCache,
        } = await buildModelFromText({
            text,
            verbose: true,
            level: 1,
            encodingSize: 20
        });

        // Act
        const output = await predictUntilEnd("The horse has evolved over the past 45 to 55", {
            vocabulary,
            wordPredictModel,
            beforeSize,
            encoderLayer, decoderLayer,
            encodeWordIndexCache,
            encodingSize: 20
        })

        // Assert
        expect(output).to.equal(text + '[END]');
    }, 10000);

    it('Should remember a couple of sentences', async function() {
        // Arrange
        const text = 'Horses are adapted to run, allowing them to quickly escape predators, and possess an excellent sense of balance and a strong fight-or-flight response. Related to this need to flee from predators in the wild is an unusual trait: horses are able to sleep both standing up and lying down, with younger horses tending to sleep significantly more than adults.';
        const {
            wordPredictModel,
            vocabulary,
            beforeSize,
            encoderLayer,
            decoderLayer,
            encodeWordIndexCache,
        } = await buildModelFromText({
            text,
            verbose: false,
            level: 1,
            encodingSize: 20
        });

        // Act
        const output = await predictUntilEnd("Horses are adapted to run, allowing them to quickly escape", {
            vocabulary,
            wordPredictModel,
            beforeSize,
            encoderLayer,
            decoderLayer,
            encodeWordIndexCache,
            encodingSize: 20,
        })

        // Assert
        expect(output).to.equal(text + '[END]');
    }, 20000);

    it.only('Should build a base model', async function() {
        // Arrange
        const encodingSize = 50;

        const {
            wordPredictModel,
            vocabulary,
            beforeSize,
            encoderDecoder,
        } = await buildModelFromText({
            text: _8Paragraphs,
            verbose: true,
            level: 2,
            epochs: 2,
            alpha: 0.0001,
            beforeSize: 30,
            encodingSize,
        });

        await serializeModel('wikiHorse', {
            wordPredictModel,
            encoderDecoder,
            vocabulary,
            beforeSize,
            encodingSize,
        });
    }, 200000);

    it.only('Should remember an entire paragraph', async function() {
        // Arrange
        const text = _1Paragraph;
        const {
            wordPredictModel,
            encoderLayer,
            decoderLayer,
            vocabulary,
            beforeSize,
            encodingSize,
            encodeWordIndexCache,
        } = await loadModel('wikiHorse');

        await trainModelWithText({
            text,
            vocabulary,
            wordPredictModel,
            verbose: true,
            beforeSize,
            encodingSize,
            encodeWordIndexCache,
            encoderLayer,
            decoderLayer,
            epochs: 4,
            alpha: 0.002
        });

        // Act
        const output = await predictUntilEnd(tokenize(text).slice(0, beforeSize).join(''), {
            vocabulary,
            wordPredictModel,
            beforeSize,
            encoderLayer,
            decoderLayer,
            encodeWordIndexCache,
            encodingSize,
        })

        // Assert
        expect(output).to.equal(text + '[END]');
    }, 300000);


    it.only('Should reload model and remember a different paragraph', async function() {
        // Arrange
        const text = otherParagraph;
        const {
            wordPredictModel,
            encoderLayer,
            decoderLayer,
            vocabulary,
            beforeSize,
            encodingSize,
            encodeWordIndexCache,
            encoderDecoder,
        } = await loadModel('wikiHorse');

        await trainModelWithText({
            text,
            vocabulary,
            wordPredictModel,
            verbose: true,
            beforeSize,
            encodingSize,
            encodeWordIndexCache,
            encoderLayer,
            decoderLayer,
            epochs: 4,
            alpha: 0.002
        });

        await serializeModel('wikiHorse2', {
            wordPredictModel,
            encoderDecoder,
            vocabulary,
            beforeSize,
            encodingSize,
        });

        // Act
        const output = await predictUntilEnd(tokenize(text).slice(0, beforeSize).join(''), {
            vocabulary,
            wordPredictModel,
            beforeSize,
            encoderLayer, decoderLayer,
            encodeWordIndexCache,
            encodingSize
        });

        // Assert
        expect(output).to.equal((text + '[END]'));
    }, 100000);

    it('Should train on 2 previously seen paragraph', async function() {
        // Arrange
        const text = _2Paragraphs;

        const {
            wordPredictModel,
            encoderLayer,
            decoderLayer,
            vocabulary,
            beforeSize,
            encodingSize,
            encodeWordIndexCache,
        } = await loadModel('wikiHorse');

        await trainModelWithText({
            text,
            vocabulary,
            wordPredictModel,
            verbose: true,
            beforeSize,
            encodingSize,
            encodeWordIndexCache,
            encoderLayer,
            decoderLayer,
            epochs: 5,
            alpha: 0.05,
        });

        // Act
        const output = await predictUntilEnd(tokenize(text).slice(0, beforeSize).join(''), {
            vocabulary,
            wordPredictModel,
            beforeSize,
            encoderLayer, decoderLayer,
            encodeWordIndexCache,
            encodingSize
        });

        // Assert
        expect(output).to.equal((text + '[END]'));
    }, 100000);


    it('Should remember 3 paragraphs', async function() {
        // Arrange
        const text = _3Paragraphs;

        const {
            wordPredictModel,
            vocabulary,
            beforeSize,
            encoderLayer,
            decoderLayer,
            encodeWordIndexCache,
        } = await buildModelFromText({
            text,
            verbose: true,
            level: 1,
            encodingSize: 50,
            epochs: 15
        });

        // Act
        const output = await predictUntilEnd(tokenize(text).slice(0, beforeSize).join(''), {
            vocabulary,
            wordPredictModel,
            beforeSize,
            encoderLayer, decoderLayer,
            encodeWordIndexCache,
            encodingSize: 50,
        });

        // Assert
        expect(output).to.equal((text + '[END]'));
    }, 100000);

    it('Should remember 4 paragraphs', async function() {
        // Arrange
        const text = _4Paragraphs;

        const {
            wordPredictModel,
            vocabulary,
            beforeSize,
            encoderLayer,
            decoderLayer,
            encodeWordIndexCache,
        } = await buildModelFromText({
            text,
            verbose: true,
            level: 1,
            encodingSize: 50,
            epochs: 15
        });

        // Act
        const output = await predictUntilEnd(tokenize(text).slice(0, beforeSize).join(''), {
            vocabulary,
            wordPredictModel,
            beforeSize,
            encoderLayer, decoderLayer,
            encodeWordIndexCache,
            encodingSize: 50,
        });

        // Assert
        expect(output).to.equal((text + '[END]'));
    }, 200000);

    it('Should remember 8 paragraphs', async function() {
        // Arrange
        const text = _8Paragraphs;

        const {
            wordPredictModel,
            vocabulary,
            beforeSize,
            encoderLayer,
            decoderLayer,
            encodeWordIndexCache,
        } = await buildModelFromText({
            text,
            verbose: true,
            level: 1,
            encodingSize: 50,
            epochs: 20
        });

        // Act
        const output = await predictUntilEnd(tokenize(text).slice(0, beforeSize).join(''), {
            vocabulary,
            wordPredictModel,
            beforeSize,
            encoderLayer, decoderLayer,
            encodeWordIndexCache,
            encodingSize: 50,
        });

        // Assert
        expect(output).to.equal((text + '[END]'));
    }, 800000);

    it('Should remember 16 paragraphs', async function() {
        // Arrange
        const text = _16Paragraphs;

        const {
            wordPredictModel,
            vocabulary,
            beforeSize,
            encoderLayer,
            decoderLayer,
            encodeWordIndexCache,
        } = await buildModelFromText({
            text,
            verbose: true,
            level: 1,
            encodingSize: 50,
            epochs: 5
        });

        // Act
        const output = await predictUntilEnd(tokenize(text).slice(0, beforeSize).join(''), {
            vocabulary,
            wordPredictModel,
            beforeSize,
            encoderLayer, decoderLayer,
            encodeWordIndexCache,
            encodingSize: 50,
        });

        // Assert
        expect(output).to.equal((text + '[END]'));
    }, 1000000);

    it.skip('Should parse and entire article and output horse information.', async function() {
        // Arrange
        const text = readFileSync(CORPUS_PATH + '/wiki-horse.txt').toString();

        const {
            wordPredictModel,
            vocabulary,
            beforeSize,
            encoderLayer,
            decoderLayer,
            encodeWordIndexCache,
        } = await buildModelFromText({
            text,
            verbose: true,
            level: 1,
            encodingSize: 100,
            epochs: 4
        });

        // Act
        const output = await predictUntilEnd(tokenize(text).slice(0, beforeSize).join(''), {
            vocabulary,
            wordPredictModel,
            beforeSize,
            encoderLayer,
            decoderLayer,
            encodeWordIndexCache,
            encodingSize: 100,
        });

        // Assert
        expect(output).to.equal((text + '[END]'));
    }, 200000);
});
