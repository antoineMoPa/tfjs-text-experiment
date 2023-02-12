import { LayersModel } from '@tensorflow/tfjs-node';

import {
    tokenize,
    buildVocabulary,
    buildTrainingData,
    buildModel,
    predict,
    predictUntilEnd,
    buildModelFromText
} from '../src/tinygpt';

import { expect } from 'chai';

describe('Model', async () => {
    it('Should build a vocabulary', async () => {
        // Arrange
        const text = 'the quick brown fox jumps over the lazy dog';

        // Act
        const vocabulary = await buildVocabulary(text);

        // Assert
        expect(vocabulary.words).to.contain('fox');
        expect(vocabulary.words).to.contain('[END]');
    });

    it('Should remember a simple word', async () => {
        // Arrange
        const text = 'the quick brown fox jumps over the lazy dog';
        const vocabulary = await buildVocabulary(text);
        const beforeSize = 3;
        const trainingData = await buildTrainingData({ vocabulary, text, beforeSize });
        const wordPredictModel = await buildModel({
            vocabulary,
            trainingData,
            verbose: false,
            beforeSize: 3,
        }) as LayersModel;

        // Act
        const { word } = await predict(
            tokenize("the quick brown"),
            { wordPredictModel, vocabulary, beforeSize}
        );

        // Assert
        expect(word).to.equal('fox');
    });

    it('Should remember a simple sentence', async () => {
        // Arrange
        const text = 'the quick brown fox jumps over the lazy dog';
        const vocabulary = await buildVocabulary(text);
        const beforeSize = 3;
        const trainingData = await buildTrainingData({ vocabulary, text, beforeSize });
        const wordPredictModel = await buildModel({
            vocabulary,
            trainingData,
            verbose: false,
            beforeSize
        }) as LayersModel;

        // Act
        const sentence = await predictUntilEnd("the quick brown", {
            vocabulary,
            wordPredictModel,
            beforeSize
        })

        // Assert
        expect(sentence).to.equal(text + ' [END]');
    });

    it('Should remember a more complex sentence', async () => {
        // Arrange
        const text = 'It belongs to the taxonomic family Equidae and is one of two extant subspecies of Equus ferus.';
        const vocabulary = await buildVocabulary(text);
        const beforeSize = 3;
        const trainingData = await buildTrainingData({ vocabulary, text, beforeSize });
        const wordPredictModel = await buildModel({
            vocabulary,
            trainingData,
            verbose: true,
            beforeSize
        }) as LayersModel;


        // Act
        const sentence = await predictUntilEnd("It belongs to", {
            vocabulary,
            wordPredictModel,
            beforeSize
        })

        // Assert
        expect(sentence).to.equal(text + ' [END]');
    });

    it('Should remember an even more complex sentence', async () => {
        // Arrange
        const text = 'The horse has evolved over the past 45 to 55 million years from a small multi-toed creature, Eohippus, into the large, single-toed animal of today';
        const { wordPredictModel, vocabulary, beforeSize } = await buildModelFromText({
            text,
            verbose: true,
            level: 0,
        });

        // Act
        const sentence = await predictUntilEnd("The horse has", {
            vocabulary,
            wordPredictModel,
            beforeSize
        })

        // Assert
        expect(sentence).to.equal(text + ' [END]');
    });

    it('Should remember a couple of sentences', async function() {
        this.timeout(10000);
        // Arrange
        const text = 'Horses are adapted to run, allowing them to quickly escape predators, and possess an excellent sense of balance and a strong fight-or-flight response. Related to this need to flee from predators in the wild is an unusual trait: horses are able to sleep both standing up and lying down, with younger horses tending to sleep significantly more than adults.';
        const { wordPredictModel, vocabulary, beforeSize } = await buildModelFromText({
            text,
            verbose: true,
            level: 1,
        });

        // Act
        const sentence = await predictUntilEnd("Horses are adapted to run", {
            vocabulary,
            wordPredictModel,
            beforeSize
        })

        // Assert
        expect(sentence).to.equal(text + ' [END]');
    });
});
