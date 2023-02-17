import { TEXT_RATER_OUTPUT } from './textRater';

export const textRaterData = {
    [TEXT_RATER_OUTPUT.REPETITIVE]: [
        'The quick the quick the quick the quick the quick the quick',
        'Microcomputers Microcomputers Microcomputers Microcomputers Microcomputers  Microcomputers  Microcomputers  Microcomputers  Microcomputers  Microcomputers',
        ' a is of a horse. a improve is a horse of a is a as a to a a disorders, of a',
        'are to to a a of a in a are to to a a of a in a.',
        'The horse has evolved over the blood of a horse is a horse or a disabled to a horse is a horse of a horse is a horse is a disabled of a horse is a horse is a disabled of a horse is a horse is a disabled of a horse is a',
        'with food, with food, with food, with food, with food, with food, with food, with food,',
        'escape predators escape predators escape predators escape predators and escape predators escape predators',
        'competition purposes competition purposes competition purposes competition purposes competition purposes competition purposes competition purposes competition purposes',
        'driving techniques driving techniques driving techniques driving techniques driving techniques driving techniques',
        'sophisticated electrical sophisticated electrical sophisticated electrical sophisticated electrical sophisticated electrical sophisticated electrical',
        'strong fight-or-flight strong fight-or-flight strong fight-or-flight strong fight-or-flight strong fight-or-flight strong fight-or-flight',
        'Early computers were Early computers were Early computers were Early computers were Early computers were',
        'meant to meant to meant to meant to meant to meant to meant to meant to',
        'only for only for only for only for only for only for only for only for'
    ],
    [TEXT_RATER_OUTPUT.GOOD]: [
        'Horses are adapted to run, allowing them to quickly escape predators, and possess an excellent sense of balance and a strong fight-or-flight response.',
        'developed from crosses between hot bloods and cold bloods, often focusing on creating breeds for specific riding purposes',
        'from which a wide variety of riding and driving techniques developed, using many different styles of',
        'Humans provide domesticated horses with food, water, and shelter as well as attention from specialists such',
        'for most competition purposes a year is added to its age each January 1 of each year in the Norther',
        'The height is expressed as the number of full hands, followed by a point, then the number of additional',
        'Early computers were meant to be used only for calculations. Simple',
        'manual instruments like the abacus have aided people in doing calculations',
        'since ancient times. Early in the Industrial Revolution, some mechanical devices',
        'were built to automate long, tedious tasks, such as guiding patterns',
        'for looms. More sophisticated electrical machines did specialized analog calculations in',
        'the early 20th century. The first digital electronic calculating machines were',
        'developed during World War II. The first semiconductor transistors in the',
        'late 1940s were followed by the silicon-based MOSFET (MOS transistor',
        ') and monolithic integrated circuit chip technologies in the late 1950s, leading',
        'to the microprocessor and the microcomputer revolution in the 1970s. The',
        'speed, power and versatility of computers have been increasing dramatically ever',
        'since then, with transistor counts increasing at a rapid pace (as',
        'The Ishango bone, a bone tool dating back to prehistoric Africa',
        'Devices have been used to aid computation for thousands of years',
        'mostly using one-to-one correspondence with fingers. The earliest counting',
        'device was most likely a form of tally stick. Later record',
        'keeping aids throughout the Fertile Crescent included calculi (clay spheres, cones',
        ', etc. which represented counts of items, likely livestock or grains, sealed',
        'in hollow unbaked clay containers.[a][4] The use of counting rods is one example.',
    ]
};

export const flatTextRaterData: string[] = Object.keys(textRaterData).map(key => {
    return textRaterData[key];
}).flat(2);
