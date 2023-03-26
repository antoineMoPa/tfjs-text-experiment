import * as tf from '@tensorflow/tfjs-node';
import { SymbolicTensor } from '@tensorflow/tfjs-node';
import { FocusLayer } from '../customLayers/focusLayer';
import { Vocabulary } from '../model';

export const denseTower = ({
    layerOutputs,
    inputs = layerOutputs,
    unitsList,
    vocabulary
}: {
    layerOutputs: tf.SymbolicTensor;
    inputs: tf.SymbolicTensor;
    unitsList: number[];
    vocabulary: Vocabulary;
}) => {
    let towerOutput = layerOutputs;
    const stages = unitsList.map((units) => {
        towerOutput = tf.layers.timeDistributed({
            layer: tf.layers.dense({
                units,
                activation: 'relu',
                kernelInitializer: tf.initializers.randomUniform({
                    minval: -0.004,
                    maxval: 0.004
                }),
            })
        }).apply(towerOutput) as SymbolicTensor;

        towerOutput = tf.layers.timeDistributed({
            layer: tf.layers.dense({
                units: inputs.shape[2],
                activation: 'relu',
                kernelInitializer: tf.initializers.randomUniform({
                    minval: -0.4,
                    maxval: 0.4
                }),
                trainable: false,
            })
        }).apply(towerOutput) as SymbolicTensor;

        towerOutput = tf.layers.multiply()
            .apply([towerOutput, inputs]) as SymbolicTensor;

        return towerOutput;
    });

    return { towerOutput, stages };
}

export const focusDenseTower = (
    {
        min,
        max,
        unitsList,
        beforeSize,
        layerOutput,
        inputs,
        vocabulary,
    } :
    {
        min: number,
        max?: number,
        unitsList: number[],
        beforeSize: number,
        layerOutput: tf.SymbolicTensor,
        inputs: tf.SymbolicTensor;
        vocabulary: Vocabulary;
    }
) => {
    return denseTower({
        layerOutputs: layerOutput as tf.SymbolicTensor,
        inputs: new FocusLayer({
            min,
            max,
            maxTimeStep: beforeSize,
        }).apply(inputs) as tf.SymbolicTensor,
        unitsList,
        vocabulary,
    });
};
