import layers_models_transforms_dicts as lmd

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

            # The first level of the dictionary must be a key called 'exps', and a value that is a list
            if 'exps' not in experiment_specs or not isinstance(experiment_specs['exps'], list):
                raise ValueError('Error: The first level of the dictionary must be a key called "exps", with a list as value')

            # Parse each experiment specification
            for exp in experiment_specs['exps']:
                # Each element in exps must have a 'name' (string) and an optional list of 'layers' (list)
                if 'name' not in exp or not isinstance(exp['name'], str):
                    raise ValueError(f'Error in experiment {exp}: Each experiment must have a key called "name", with a string as value')

                if 'layers' in exp and not isinstance(exp['layers'], list):
                    raise ValueError(f'Error in experiment {exp}: "layers" must have a list as value')

                # Each experiment must have a valid 'base_model' (defined in models_dict)
                if 'base_model' not in exp or exp['base_model'] not in lmd.MODELS_DICT:
                    raise ValueError(f'Error in experiment {exp}: Each experiment must have a key called "base_model", whose value is defined in models_dict')

                # Each experiment must have a valid 'output_format' (defined in transformers_dict)
                if 'output_format' not in exp or exp['output_format'] not in lmd.TRANSFORMERS_DICT:
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
                    if 'layer' not in layer or layer['layer'] not in lmd.LAYERS_DICT:
                        raise ValueError(f'Error in experiment {exp["name"]}, layer {layer}: Each layer must have a key called "layer", whose value is defined in layers_dict')

                    if 'args' in layer and not isinstance(layer['args'], list):
                        raise ValueError(f'Error in experiment {exp["name"]}, layer {layer["layer"]}: "args" must have a list as value')

                    # Create default empty argument list if args was not defined. This way, args will not have to be specified if it is empty
                    if 'args' not in layer:
                        layer.update({'args': []})

                    # Each element of args must be a dictionary of a single {value: data type} (data_type must be a valid type defined in TYPE_CONVERTERS)
                    for arg in layer['args']:
                        if not isinstance(arg, dict) or len(arg) != 1 or list(arg.values())[0] not in lmd.TYPE_CONVERTERS:
                            raise ValueError(f'Error in experiment {exp["name"]}, layer {layer["layer"]}, argument {arg}: Each layer argument must be a dict with a single {{value: data-type}}, and data-type must be in TYPE_CONVERTERS')

                    if 'kwargs' in layer and not isinstance(layer['kwargs'], dict):
                        raise ValueError(f'Error in experiment {exp["name"]}, layer {layer["layer"]}: "kwargs" must have a dict as value')

                    # Create default empty kw-argument dict if kwargs was not defined. This way, kwargs will not have to be specified if it is empty
                    if 'kwargs' not in layer:
                        layer.update({'kwargs': {}})

                    # Kwargs must be a dictionary, with format {kwarg_name: {value: data type}} (data_type must be a valid type defined in TYPE_CONVERTERS)
                    for kwarg, arg in zip(layer['kwargs'].keys(), layer['kwargs'].values()):
                        if not isinstance(arg, dict) or len(arg) != 1 or list(arg.values())[0] not in lmd.TYPE_CONVERTERS:
                            raise ValueError(f'Error in experiment {exp["name"]}, layer {layer["layer"]}, kw-argument {kwarg}: Each layer kw-argument must be a dict with a single {{value: data-type}}, and data-type must be in TYPE_CONVERTERS')
                    

            # Everything should be OK now. Parameter errors are not responsibility of the parser.
            return experiment_specs

        except ValueError as e:
            print(e)
            return {}