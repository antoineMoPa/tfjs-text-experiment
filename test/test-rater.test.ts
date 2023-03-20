import { buildVocabulary } from '../src/model';
import { buildEncoderDecoder } from '../src/encoderDecoder';
import {
    buildTextRater,
    TEXT_RATER_OUTPUT,
    rateText,
    TEXT_RATER_OUTPUT_LEN,
} from '../src/textRater';
import { describe, it, expect } from 'vitest';

import {
    flatTextRaterData,
    validationData
} from '../src/textRaterData';

describe('Text Rater', async () => {
    it('Rates text', async function () {
        // Arrange
        const encodingSize = 45;
        const text1 = 'Early computers were meant to be used only for calculations.';
        const text2 = 'quick quick quick quick quick quick quick quick quick quick';
        const vocabulary = buildVocabulary(...[text1, text2, ...flatTextRaterData]);

        const { encoderLayer } = await buildEncoderDecoder({ vocabulary, encodingSize })
        const { textRater } = await buildTextRater({
            vocabulary,
            encoderLayer,
            encodingSize,
        });

        // Act

        for (let i = 0; i < TEXT_RATER_OUTPUT_LEN; i++) {
            let success = 0;
            let total = 0;

            validationData[i].forEach(text => {
                const result = rateText(
                    text,
                    {
                        vocabulary,
                        encoderLayer,
                        textRater,
                        encodingSize
                    }
                );

                result === i && success++;
                total++;
            });

            const expectedSuccessRate = {
                [TEXT_RATER_OUTPUT.GOOD]: 0.3,
                [TEXT_RATER_OUTPUT.REPETITIVE]: 0.8
            }

            if (expectedSuccessRate[i]) {
                expect(success/total).to.be.greaterThan(expectedSuccessRate[i]);
            }
        }
    }, 50000);
});
