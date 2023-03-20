import * as tf from '@tensorflow/tfjs-node';
import { SymbolicTensor } from '@tensorflow/tfjs-node';
import { FocusLayer } from '../customLayers/focusLayer';

export const denseTower = ({
    layerOutputs,
    inputs = layerOutputs,
    unitsList
}: {
    layerOutputs: tf.SymbolicTensor;
    inputs: tf.SymbolicTensor;
    unitsList: number[];
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
                biasInitializer: tf.initializers.constant({ value: -0.02 }),
            })
        }).apply(towerOutput) as SymbolicTensor;

        towerOutput = tf.layers.layerNormalization().apply(towerOutput) as SymbolicTensor;

        towerOutput = tf.layers.concatenate().apply([
            inputs,
            towerOutput,
        ]) as SymbolicTensor;

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
    return denseTower({
        layerOutputs: layerOutput as tf.SymbolicTensor,
        inputs: new FocusLayer({
            min,
            max,
            maxTimeStep: beforeSize,
        }).apply(inputs) as tf.SymbolicTensor,
        unitsList
    });
};
