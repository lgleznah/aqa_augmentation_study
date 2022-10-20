import valid_parameters_dicts as vpd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer

from keras_cv.layers.preprocessing import MaybeApply

def get_augmented_model_and_preprocess(experiment_specs, seed):
    layer_list = []

    # Fetch model parameters: base model, input shape, and the specific preprocessing function of the model
    base_model_func, input_shape, preprocess_func = vpd.MODELS_DICT[experiment_specs['base_model']]

    # Fetch base model, pre-trained on ImageNet; and compute the number of output neurons
    base_model = base_model_func(input_shape=input_shape, include_top=False, pooling ="avg", weights='imagenet')
    _, _, output_activation, output_neurons = vpd.TRANSFORMERS_DICT[experiment_specs['output_format']]

    # Create all augmentation layers
    for layer in experiment_specs['layers']:

        # Parse layer arguments, converting them to their correct type
        layer_args = []
        for argument in layer['args']:
            argument_type = list(argument.values())[0]
            argument_value = list(argument.keys())[0]
            type_converter = vpd.TYPE_CONVERTERS[argument_type]
            layer_args.append(type_converter(argument_value))

        # Parse layer keyword arguments, converting them to their correct type
        layer_kwargs = {}
        for kwarg in layer['kwargs'].items():
            kwarg_name = kwarg[0]
            kwarg_type = list(kwarg[1].values())[0]
            kwarg_value = list(kwarg[1].keys())[0]
            type_converter = vpd.TYPE_CONVERTERS[kwarg_type]
            layer_kwargs[kwarg_name] = type_converter(kwarg_value)
        
        layer_kwargs.update({'seed': seed})

        # Create layer with the given arguments, and add to the list of layers
        # If this layer has an augmentation probability between 0 and 1, then wrap
        # the layer with MaybeApply
        new_layer = vpd.LAYERS_DICT[layer['layer']](*layer_args, **layer_kwargs)
        if (float(layer['rate']) < 1.0):
            new_layer = MaybeApply(layer=new_layer, rate=float(layer['rate']))

        layer_list.append(new_layer)
    
    return Sequential([InputLayer(input_shape=input_shape)] + layer_list + [base_model, Dense(output_neurons, activation=output_activation)]), preprocess_func