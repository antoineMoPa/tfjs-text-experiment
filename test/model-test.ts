import { LayersModel } from '@tensorflow/tfjs-node';
import { readFileSync } from 'fs';

import {
    tokenize,
    buildVocabulary,
    buildTrainingData,
    buildModel,
    predict,
    predictUntilEnd,
    buildModelFromText,
    CORPUS_PATH
} from '../src/tinygpt';

import { expect } from 'chai';

import { twoParagraphs } from './testText';

describe.only('Model', async () => {
    it.skip('Should build a vocabulary', async () => {
        // Arrange
        const text = 'the quick brown fox jumps over the lazy dog';

        // Act
        const vocabulary = buildVocabulary(text);

        // Assert
        expect(vocabulary.words).to.contain(' fox');
        expect(vocabulary.words).to.contain('[END]');
    });

    it.skip('Should remember a simple word', async function () {
        this.timeout(10000)
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
            encodingSize: 7
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
    });

    it('Should remember a simple sentence', async function () {
        this.timeout(10000)
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
    });

    it('Should remember a more complex sentence', async function () {
        this.timeout(10000);

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
    });

    it('Should remember an even more complex sentence', async function () {
        this.timeout(10000);

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
        const sentence = await predictUntilEnd("The horse has evolved over", {
            vocabulary,
            wordPredictModel,
            beforeSize,
            encoderLayer, decoderLayer,
            encodeWordIndexCache,
            encodingSize: 20
        })

        // Assert
        expect(sentence).to.equal(text + '[END]');
    });

    it('Should remember a couple of sentences', async function() {
        this.timeout(10000);

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
        const sentence = await predictUntilEnd("Horses are adapted to run,", {
            vocabulary,
            wordPredictModel,
            beforeSize,
            encoderLayer,
            decoderLayer,
            encodeWordIndexCache,
            encodingSize: 20,
        })

        // Assert
        expect(sentence).to.equal(text + '[END]');
    });

    it('Should remember an entire paragraph', async function() {
        this.timeout(20000);

        // Arrange
        const text = 'Horses and humans interact in a wide variety of sport competitions and non-competitive recreational pursuits as well as in working activities such as police work, agriculture, entertainment, and therapy. Horses were historically used in warfare, from which a wide variety of riding and driving techniques developed, using many different styles of equipment and methods of control. Many products are derived from horses, including meat, milk, hide, hair, bone, and pharmaceuticals extracted from the urine of pregnant mares. Humans provide domesticated horses with food, water, and shelter as well as attention from specialists such as veterinarians and farriers.';

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
            encodingSize: 30
        });

        // Act
        const sentence = await predictUntilEnd("Horses and humans interact in a wide", {
            vocabulary,
            wordPredictModel,
            beforeSize,
            encoderLayer,
            decoderLayer,
            encodeWordIndexCache,
            encodingSize: 30,
        })

        // Assert
        expect(sentence).to.equal(text + '[END]');
    });

    it('Should remember multiple paragraphs', async function() {
        this.timeout(50000);
        // Arrange
        const text = twoParagraphs;

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
        });

        // Act
        const sentence = await predictUntilEnd("Horse breeds are loosely divided into", {
            vocabulary,
            wordPredictModel,
            beforeSize,
            encoderLayer, decoderLayer,
            encodeWordIndexCache,
            encodingSize: 50,
        });

        // Assert
        expect(sentence).to.equal((text + '[END]'));
    });

    it.skip('Should parse and entire article and output horse information.', async function() {
        this.timeout(40000);
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
            level: 2,
            encodingSize: 128,
        });

        // Act
        const sentence = await predictUntilEnd("Horse breeds are loosely divided into", {
            vocabulary,
            wordPredictModel,
            beforeSize,
            encoderLayer,
            decoderLayer,
            encodeWordIndexCache,
            encodingSize: 128,
        })

        // Assert
        expect(sentence).to.equal((text + '[END]'));
    });
});
