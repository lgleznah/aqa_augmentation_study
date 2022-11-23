import sys

def main():
    filename = sys.argv[1]
    intensity = sys.argv[2]
    rates = [float(rate) for rate in sys.argv[3:]]

    brightness_template = (
        '  - name: brightness_{rate_pct}\n'
        '    base_model: mobilenet\n'
        '    output_format: distribution\n'
        '    batch_size: 128\n'
        '    layers:\n'
        '      - layer: brightness\n'
        '        rate: {rate}\n'
        '        args:\n'
        '          - {intensity}: float\n'
        '        kwargs:\n'
        '          value_range:\n' 
        '            (-1,1): tuple\n'
    )

    contrast_template = (
        '  - name: contrast_{rate_pct}\n'
        '    base_model: mobilenet\n'
        '    output_format: distribution\n'
        '    batch_size: 128\n'
        '    layers:\n'
        '      - layer: contrast\n'
        '        rate: {rate}\n'
        '        args:\n'
        '          - {intensity}: float\n'
    )

    flip_template = (
        '  - name: flip_{rate_pct}\n'
        '    base_model: mobilenet\n'
        '    output_format: distribution\n'
        '    batch_size: 128\n'
        '    layers:\n'
        '      - layer: flip\n'
        '        rate: {rate}\n'
        '        args:\n'
        '          - horizontal: string\n'
    )

    rotation_template = (
        '  - name: rotation_{rate_pct}\n'
        '    base_model: mobilenet\n'
        '    output_format: distribution\n'
        '    batch_size: 128\n'
        '    layers:\n'
        '      - layer: rotation\n'
        '        rate: {rate}\n'
        '        args:\n'
        '          - {intensity}: float\n'
        '        kwargs:\n'
        '          fill_mode:\n'
        '            constant: string\n'
    )

    translation_template = (
        '  - name: translation_{rate_pct}\n'
        '    base_model: mobilenet\n'
        '    output_format: distribution\n'
        '    batch_size: 128\n'
        '    layers:\n'
        '      - layer: translation\n'
        '        rate: {rate}\n'
        '        args:\n'
        '          - {intensity}: float\n'
        '          - {intensity}: float\n'
        '        kwargs:\n'
        '          fill_mode:\n'
        '            constant: string\n'
    )

    zoom_template = (
        '  - name: zoom_{rate_pct}\n'
        '    base_model: mobilenet\n'
        '    output_format: distribution\n'
        '    batch_size: 128\n'
        '    layers:\n'
        '      - layer: zoom\n'
        '        rate: {rate}\n'
        '        args:\n'
        '          - {intensity}: float\n'
        '        kwargs:\n'
        '          width_factor:\n' 
        '            {intensity}: float\n'
        '          fill_mode:\n'
        '            constant: string\n'
    )

    templates = [brightness_template, contrast_template, flip_template, rotation_template, translation_template, zoom_template]

    with open(filename, 'w') as f:
        for template in templates:
            for rate in rates:
                print(template.format(rate_pct = int(rate*100), rate = rate, intensity = intensity), file=f)


if __name__ == '__main__':
    main()