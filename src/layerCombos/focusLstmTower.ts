import * as tf from '@tensorflow/tfjs-node';
import { SymbolicTensor } from '@tensorflow/tfjs-node';
import { FocusLayer } from '../customLayers/focusLayer';

export const lstmTower = ({
    layerOutputs,
    inputs = layerOutputs,
    unitsList
}: {
    layerOutputs: tf.SymbolicTensor;
    inputs: tf.SymbolicTensor;
    unitsList: number[];
}) => {
    let towerOutput = layerOutputs;
    const stages = unitsList.map((units, index) => {

        const dense = () => {
            towerOutput = tf.layers.timeDistributed({
                layer: tf.layers.dense({
                    units,
                    activation: 'relu',
                    kernelInitializer: tf.initializers.randomUniform({
                        minval: -0.01,
                        maxval: 0.01
                    }),
                    biasInitializer: tf.initializers.constant({ value: -0.01 }),
                })
            }).apply(towerOutput) as SymbolicTensor;
        };

        dense();

        const lstmOutput = tf.layers.lstm({
            units,
            activation: 'relu',
            returnSequences: true,
            kernelInitializer: tf.initializers.randomUniform({
                minval: -0.002,
                maxval: 0.002
            }),
            recurrentInitializer: tf.initializers.randomUniform({
                minval: -0.002,
                maxval: 0.002
            }),
            biasInitializer: tf.initializers.constant({ value: -0.01 }),
            dropout: 0.01,
            recurrentDropout: 0.01,
        }).apply(towerOutput) as SymbolicTensor;

        towerOutput = lstmOutput;

        towerOutput = tf.layers.concatenate().apply([
            inputs,
            towerOutput,
        ]) as SymbolicTensor;

        return towerOutput;
    });

    return { towerOutput, stages };
}

export const focusedLstmTower = (
    {
        min,
        max,
        unitsList,
        beforeSize,
        layerOutput,
        inputs
    } :
    {
        min: number,
        max?: number,
        unitsList: number[],
        beforeSize: number,
        layerOutput: tf.SymbolicTensor,
        inputs: tf.SymbolicTensor
    }
) => {
    return lstmTower({
        layerOutputs: new FocusLayer({
            min,
            max,
            maxTimeStep: beforeSize,
        }).apply(layerOutput) as tf.SymbolicTensor,
        inputs: new FocusLayer({
            min,
            max,
            maxTimeStep: beforeSize,
        }).apply(inputs) as tf.SymbolicTensor,
        unitsList
    }).towerOutput;
};
