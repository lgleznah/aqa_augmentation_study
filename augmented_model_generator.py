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

    # TODO: make this mess more legible
    for layer in experiment_specs['layers']:
        layer_args = [ lmd.TYPE_CONVERTERS[list(value_type.values())[0]](list(value_type.keys())[0]) for value_type in layer['args'] ]
        layer_kwargs = { kwarg[0]: lmd.TYPE_CONVERTERS[list(kwarg[1].values())[0]](list(kwarg[1].keys())[0]) for kwarg in layer['kwargs'].items() }
        new_layer = lmd.LAYERS_DICT[layer['layer']](*layer_args, **layer_kwargs)
        layer_list.append(new_layer)
    
    return Sequential([InputLayer(input_shape=input_shape)] + layer_list + [base_model, Dense(output_neurons, activation=output_activation)]), preprocess_func