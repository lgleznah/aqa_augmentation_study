import valid_parameters_dicts as vpd

import yaml

def parse_experiment_file(filename):
    """
    Parses a file containing a set of data augmentation experiments.
    
    :param filename: the name of the YAML file containing the experiments.
    :return a dictionary containing all the experiments, or an empty dictionary if the parsing was unsuccessful
    """
    with open(filename, 'r') as f:
        try:
            experiment_specs = yaml.safe_load(f)

            # One of the top-level entries of the dictionary must be a key called 'seed', to set the random seed for the experiments.
            # The value must be an integer
            if 'seed' not in experiment_specs or not isinstance(experiment_specs['seed'], int):
                raise ValueError('Error: The dictionary must contain a top-level key called "seed", with an int as value')

            # Verify that there is also a number of epochs set
            if 'epochs' not in experiment_specs or not isinstance(experiment_specs['epochs'], int):
                raise ValueError('Error: The dictionary must contain a top-level key called "epochs", with an int as value')

            # The other top-level entry of the dictionary must be a key called 'exps', and a value that is a list
            if 'exps' not in experiment_specs or not isinstance(experiment_specs['exps'], list):
                raise ValueError('Error: The dictionary must contain a top-level key called "exps", with a list as value')

            # The last top-level entry of the dictionary must be a key called 'dataset', and a value that is a string (defined in vpd.DATASETS_DICTS)
            if 'dataset' not in experiment_specs or not isinstance(experiment_specs['dataset'], str) or experiment_specs['dataset'] not in vpd.DATASETS_DICT:
                raise ValueError('Error: The dictionary must contain a top-level key called "dataset", with a string as value. The string must be defined in vpd.DATASETS_DICT')

            # Another top-level entry must be a key called "lr", whose value is a float
            if 'lr' not in experiment_specs or not isinstance(experiment_specs['lr'], float):
                raise ValueError('Error: The dictionary must contain a top-level key called "lr", with a float as value')

            # Another top-level entry must be a key called "use_plateau", whose value is a boolean
            if 'use_plateau' not in experiment_specs or not isinstance(experiment_specs['use_plateau'], bool):
                raise ValueError('Error: The dictionary must contain a top-level key called "use_plateau", with a bool as value')

            # Another top-level entry must be a key called "weight_classes", whose value is a boolean
            if 'weight_classes' not in experiment_specs or not isinstance(experiment_specs['weight_classes'], bool):
                raise ValueError('Error: The dictionary must contain a top-level key called "weight_classes", with a bool as value')

            # Optional argument: validation split size. Must be a float between 0 and 1
            if 'val_split' not in experiment_specs:
                experiment_specs.update({'val_split': 0.2})

            if 'val_split' in experiment_specs and (not isinstance(experiment_specs['val_split'], float) or (experiment_specs['val_split'] < 0 or experiment_specs['val_split'] > 1)):
                raise ValueError('Error: val_split must be a float between 0 and 1')
            
            # Optional argument: test split size. Must be a float between 0 and 1
            if 'test_split' not in experiment_specs:
                experiment_specs.update({'test_split': 0.08})

            if 'test_split' in experiment_specs and (not isinstance(experiment_specs['test_split'], float) or (experiment_specs['test_split'] < 0 or experiment_specs['test_split'] > 1)):
                raise ValueError('Error: test_split must be a float between 0 and 1')

            # Parse each experiment specification
            for exp in experiment_specs['exps']:
                # Each element in exps must have a 'name' (string) and an optional list of 'layers' (list)
                if 'name' not in exp or not isinstance(exp['name'], str):
                    raise ValueError(f'Error in experiment {exp}: Each experiment must have a key called "name", with a string as value')

                if 'layers' in exp and not isinstance(exp['layers'], list):
                    raise ValueError(f'Error in experiment {exp}: "layers" must have a list as value')

                # Each experiment must have a valid 'base_model' (defined in models_dict)
                if 'base_model' not in exp or exp['base_model'] not in vpd.MODELS_DICT:
                    raise ValueError(f'Error in experiment {exp}: Each experiment must have a key called "base_model", whose value is defined in models_dict')

                # Each experiment must have a valid 'output_format' (defined in transformers_dict)
                if 'output_format' not in exp or exp['output_format'] not in vpd.TRANSFORMERS_DICT:
                    raise ValueError(f'Error in experiment {exp}: Each experiment must have a key called "output_format", whose value is defined in transformers_dict')

                # Each experiment must have a 'batch_size' (int)
                if 'batch_size' not in exp or not isinstance(exp['batch_size'], int):
                    raise ValueError(f'Error in experiment {exp}: Each experiment must have a key called "batch_size", with an int as value')

                # Add empty layer list if layers was not specified
                if 'layers' not in exp:
                    exp.update({'layers': []})

                # Parse each layer
                for layer in exp['layers']:
                    # Each element in layers must have a valid 'layer' (defined in layers_dict).
                    # Optionally a list of 'args', and a dict of 'kwargs'
                    if 'layer' not in layer or layer['layer'] not in vpd.LAYERS_DICT:
                        raise ValueError(f'Error in experiment {exp["name"]}, layer {layer}: Each layer must have a key called "layer", whose value is defined in layers_dict')

                    if 'args' in layer and not isinstance(layer['args'], list):
                        raise ValueError(f'Error in experiment {exp["name"]}, layer {layer["layer"]}: "args" must have a list as value')

                    # Optionally, each layer can specify the probability of its inputs being augmented, as a float between 0 and 1
                    # If this value is omitted, it is given a default value of 1.0
                    if 'rate' in layer and (not isinstance(layer['rate'], float) or layer['rate'] < 0.0 or layer['rate'] > 1.0):
                        raise ValueError(f'Error in experiment {exp["name"]}, layer {layer["layer"]}: "rate" must be a float between 0 and 1')

                    if 'rate' not in layer:
                        layer.update({'rate': 1.0})

                    # Create default empty argument list if args was not defined. This way, args will not have to be specified if it is empty
                    if 'args' not in layer:
                        layer.update({'args': []})

                    # Each element of args must be a dictionary of a single {value: data type} (data_type must be a valid type defined in TYPE_CONVERTERS)
                    for arg in layer['args']:
                        if not isinstance(arg, dict) or len(arg) != 1 or list(arg.values())[0] not in vpd.TYPE_CONVERTERS:
                            raise ValueError(f'Error in experiment {exp["name"]}, layer {layer["layer"]}, argument {arg}: Each layer argument must be a dict with a single {{value: data-type}}, and data-type must be in TYPE_CONVERTERS')

                    if 'kwargs' in layer and not isinstance(layer['kwargs'], dict):
                        raise ValueError(f'Error in experiment {exp["name"]}, layer {layer["layer"]}: "kwargs" must have a dict as value')

                    # Create default empty kw-argument dict if kwargs was not defined. This way, kwargs will not have to be specified if it is empty
                    if 'kwargs' not in layer:
                        layer.update({'kwargs': {}})

                    # Kwargs must be a dictionary, with format {kwarg_name: {value: data type}} (data_type must be a valid type defined in TYPE_CONVERTERS)
                    for kwarg, arg in zip(layer['kwargs'].keys(), layer['kwargs'].values()):
                        if not isinstance(arg, dict) or len(arg) != 1 or list(arg.values())[0] not in vpd.TYPE_CONVERTERS:
                            raise ValueError(f'Error in experiment {exp["name"]}, layer {layer["layer"]}, kw-argument {kwarg}: Each layer kw-argument must be a dict with a single {{value: data-type}}, and data-type must be in TYPE_CONVERTERS')
                    

            # Everything should be OK now. Parameter errors are not responsibility of the parser.
            return experiment_specs

        except ValueError as e:
            print(e)
            return {}