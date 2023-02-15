import { tokenize } from '../tinygpt';
import { sum } from 'lodash';

// Honestly, I'm not sure yet how to use that
export function textEntropy(text: string, vocabulary: string[]): number {
    const tokens = tokenize(text);
    const wordOccurences = new Map<string, number>();

    vocabulary.forEach(token => wordOccurences.set(token, 0));

    tokens.forEach(token => {
        wordOccurences.set(token, wordOccurences.get(token) + 1);
    });

    const tokenProbabilities = tokens.map(token => wordOccurences.get(token) / tokens.length);
    const entropies = tokenProbabilities.map(p => p * Math.log(p));

    return -sum(entropies);
}
