import {
    textEntropy
} from '../src/metrics/entropy';
import {
    buildVocabulary
} from '../src/model';

import { describe, it, expect } from 'vitest';

describe('Metrics', async () => {
    it('Should calculate entropy', async () => {
        // Arrange
        const text = 'The horse has evolved over the three categories based';
        const vocabulary = buildVocabulary(text);

        // Act
        const entropy = textEntropy(text, vocabulary);

        // Assert
        expect(entropy).to.be.a('number');
    });
});
