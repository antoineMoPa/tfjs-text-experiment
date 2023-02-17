import * as tf from '@tensorflow/tfjs-node';

import { buildVocabulary } from '../src/tinygpt';
import { buildEncoderDecoder } from '../src/encoderDecoder';
import {
    buildTextRater,
    TEXT_RATER_OUTPUT,
    rateText
} from '../src/textRater';
import { expect } from 'chai';
import { flatTextRaterData, textRaterData } from '../src/textRaterData';

describe.only('Text Rater', async () => {
    it('Rates text', async function () {
        this.timeout(30000);

        // Arrange
        const encodingSize = 10;
        const text1 = 'Early computers were meant to be used only for calculations.';
        const text2 = 'quick quick quick quick quick quick quick quick quick quick';
        const vocabulary = await buildVocabulary(...[text1, text2, ...flatTextRaterData]);
        const { encoderLayer } = await buildEncoderDecoder({ vocabulary, encodingSize })
        const { textRater } = await buildTextRater({
            vocabulary,
            inputSize: 10,
            encodingSize,
            encoderLayer
        });

        // Act

        for (let i = 0; i < Object.keys(textRaterData).length; i++) {
            textRaterData[i].forEach(text => {
                console.log(`Should be ${i}`);
                const result1 = rateText(
                    text,
                    {
                        vocabulary,
                        encoderLayer,
                        textRater,
                    }
                );
            });
        }


        const result1 = rateText(
            text1,
            {
                vocabulary,
                encoderLayer,
                textRater,
            }
        );

        const result2 = rateText(
            text2,
            {
                vocabulary,
                encoderLayer,
                textRater,
            }
        );

        // Assert
        expect(result1, 'Should be GOOD').to.equal(TEXT_RATER_OUTPUT.GOOD);
        expect(result2, 'Should be REPETITIVE').to.equal(TEXT_RATER_OUTPUT.REPETITIVE);
    });
});
