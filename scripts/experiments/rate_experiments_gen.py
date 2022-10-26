import sys

def main():
    filename = sys.argv[1]
    rates = [float(rate) for rate in sys.argv[2:]]

    brightness_template = (
        '  - name: brightness_{}\n'
        '    base_model: mobilenet\n'
        '    output_format: distribution\n'
        '    batch_size: 128\n'
        '    layers:\n'
        '      - layer: brightness\n'
        '        rate: {}\n'
        '        args:\n'
        '          - 0.5: float\n'
        '        kwargs:\n'
        '          value_range:\n' 
        '            (-1,1): tuple\n'
    )

    contrast_template = (
        '  - name: contrast_{}\n'
        '    base_model: mobilenet\n'
        '    output_format: distribution\n'
        '    batch_size: 128\n'
        '    layers:\n'
        '      - layer: contrast\n'
        '        rate: {}\n'
        '        args:\n'
        '          - 0.5: float\n'
    )

    flip_template = (
        '  - name: flip_{}\n'
        '    base_model: mobilenet\n'
        '    output_format: distribution\n'
        '    batch_size: 128\n'
        '    layers:\n'
        '      - layer: flip\n'
        '        rate: {}\n'
    )

    rotation_template = (
        '  - name: rotation_{}\n'
        '    base_model: mobilenet\n'
        '    output_format: distribution\n'
        '    batch_size: 128\n'
        '    layers:\n'
        '      - layer: rotation\n'
        '        rate: {}\n'
        '        args:\n'
        '          - 1.0: float\n'
    )

    translation_template = (
        '  - name: translation_{}\n'
        '    base_model: mobilenet\n'
        '    output_format: distribution\n'
        '    batch_size: 128\n'
        '    layers:\n'
        '      - layer: translation\n'
        '        rate: {}\n'
        '        args:\n'
        '          - 0.5: float\n'
        '          - 0.5: float\n'
    )

    zoom_template = (
        '  - name: zoom_{}\n'
        '    base_model: mobilenet\n'
        '    output_format: distribution\n'
        '    batch_size: 128\n'
        '    layers:\n'
        '      - layer: zoom\n'
        '        rate: {}\n'
        '        args:\n'
        '          - 0.5: float\n'
        '        kwargs:\n'
        '          width_factor:\n' 
        '            0.5: float\n'
    )

    templates = [brightness_template, contrast_template, flip_template, rotation_template, translation_template, zoom_template]

    with open(filename, 'w') as f:
        for template in templates:
            for rate in rates:
                print(template.format(int(rate*100), rate), file=f)


if __name__ == '__main__':
    main()