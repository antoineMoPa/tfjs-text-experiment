import { LayersModel } from '@tensorflow/tfjs-node';

import {
    tokenize,
    buildVocabulary,
    buildTrainingData,
    buildModel,
    predict,
    predictUntilEnd
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
        const trainingData = await buildTrainingData({ vocabulary, text });
        const wordPredictModel = await buildModel({ vocabulary, trainingData, verbose: false }) as LayersModel;

        // Act
        const { word } = await predict(
            tokenize("the quick brown"),
            wordPredictModel,
            vocabulary
        );

        // Assert
        expect(word).to.equal('fox');
    });

    it('Should remember a simple sentence', async () => {
        // Arrange
        const text = 'the quick brown fox jumps over the lazy dog';
        const vocabulary = await buildVocabulary(text);
        const trainingData = await buildTrainingData({ vocabulary, text });
        const wordPredictModel = await buildModel({ vocabulary, trainingData, verbose: false }) as LayersModel;

        // Act
        const sentence = await predictUntilEnd("the quick brown", {
            vocabulary,
            wordPredictModel
        })

        // Assert
        expect(sentence).to.equal(text + ' [END]');
    });
});
