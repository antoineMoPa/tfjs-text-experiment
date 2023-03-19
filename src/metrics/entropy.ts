import { tokenize, Vocabulary } from '../model';
import { sum } from 'lodash';

type TokenProbabilities = {
    token: string;
    probability: number;
}[];

export function tokenProbabilities(text: string, vocabulary: Vocabulary): TokenProbabilities {
    const tokens = tokenize(text);
    const wordOccurences = new Map<string, number>();

    vocabulary.words.forEach(token => wordOccurences.set(token, 0));

    tokens.forEach(token => {
        wordOccurences.set(token, wordOccurences.get(token) + 1);
    });

    return tokens
        .map(token => ({
            token,
            probability: wordOccurences.get(token) / tokens.length
        }))
        .sort((a, b) =>
            a.probability < b.probability ? -1 :
            a.probability ===  b.probability ? 0 : 1);
}


// Honestly, I'm not sure yet how to use that
export function textEntropy(text: string, vocabulary: Vocabulary): number {
    const tokens = tokenize(text);
    const wordOccurences = new Map<string, number>();

    vocabulary.words.forEach(token => wordOccurences.set(token, 0));

    tokens.forEach(token => {
        wordOccurences.set(token, wordOccurences.get(token) + 1);
    });

    const probs = tokenProbabilities(text, vocabulary);
    const entropies = probs.map(p => p.probability * Math.log(p.probability));

    return -sum(entropies);
}
