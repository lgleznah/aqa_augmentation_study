import layers_models_transforms_dicts as lmd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer

def get_augmented_model_and_preprocess(experiment_specs):
    layer_list = []

    # Fetch model parameters: base model, input shape, and the specific preprocessing function of the model
    base_model_func, input_shape, preprocess_func = lmd.MODELS_DICT[experiment_specs['base_model']]

    # Fetch base model, pre-trained on ImageNet; and compute the number of output neurons
    base_model = base_model_func(input_shape=input_shape, include_top=False, pooling ="avg", weights='imagenet')
    _, _, output_activation, output_neurons = lmd.TRANSFORMERS_DICT[experiment_specs['output_format']]

    for layer in experiment_specs['layers']:
        new_layer = lmd.LAYERS_DICT[layer['layer']](*layer['args'], **layer['kwargs'])
        layer_list.append(new_layer)
    
    return Sequential([InputLayer(input_shape=input_shape)] + layer_list + [base_model, Dense(output_neurons, activation=output_activation)]), preprocess_func