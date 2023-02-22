import { buildVocabulary } from '../src/tinygpt';
import { buildEncoderDecoder } from '../src/encoderDecoder';
import {
    buildTextRater,
    TEXT_RATER_OUTPUT,
    rateText,
    TEXT_RATER_OUTPUT_LEN,
} from '../src/textRater';
import { expect } from 'chai';
import { textRaterData } from '../src/textRaterData';

describe('Text Rater', async () => {
    it('Rates text', async function () {
        this.timeout(50000);

        // Arrange
        const text1 = 'Early computers were meant to be used only for calculations.';
        const text2 = 'quick quick quick quick quick quick quick quick quick quick';
        const { textRater } = await buildTextRater();

        // Act

        for (let i = 0; i < TEXT_RATER_OUTPUT_LEN; i++) {
            let success = 0;
            let total = 0;

            textRaterData[i].forEach(text => {
                const result = rateText(
                    text,
                    {
                        textRater,
                    }
                );

                result === i && success++;
                total++;
            });

            console.log(`success rate(class ${TEXT_RATER_OUTPUT[i]}): ${success}/${total} ${(success/total*100.0).toFixed(0)}%`);
        }

        const result1 = rateText(
            text1,
            {
                textRater,
            }
        );

        const result2 = rateText(
            text2,
            {
                textRater,
            }
        );

        // Assert
        expect(result1, 'Should be GOOD').to.equal(TEXT_RATER_OUTPUT.GOOD);
        expect(result2, 'Should be REPETITIVE').to.equal(TEXT_RATER_OUTPUT.REPETITIVE);
    });
});
