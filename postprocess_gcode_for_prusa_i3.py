# Annotates gcode file to add a color change pause on
# Prusa i3 printers at a print height (default 7mm).
# Usage:
# python postprocess_gcode.py /path/to/file.gcode [layer_height_in_mm]

import sys

color_change_gcode = [
    ';COLOR_CHANGE\n',
    'M600\n',
    'G1 E0.3 F1500 ; prime after color change\n',
    'M204 S800\n',
]

path = sys.argv[1]
switch_height = sys.argv.get(2) if len(sys.argv) > 2 else 7

with open(path, 'r+') as gcode_file:
    prev_line = ''
    mod_index = -1
    lines = gcode_file.readlines()
    for index, line in enumerate(lines):
        if ';LAYER_CHANGE' not in prev_line:
            prev_line = line
            continue
        layer_height = float(line.strip().split(':')[1])
        if layer_height <= switch_height:
            prev_line = line
            continue
        
        print('Adding color change pause at %fmm height' % layer_height)
        mod_index = index

        # Fast-forward past layer-change wipes to insert color change just
        # before the layer actually begins printing.
        while mod_index < len(lines) - 1:
            mod_index += 1
            if ';TYPE:Perimeter' in lines[mod_index]:
                # Adjustment of acceleration rate (M204 command) appears to
                # precede layer printing, so go back two steps.
                mod_index -= 2
                break
        break

    if mod_index == -1:
        print('Could not find appropriate location for color change insertion')
    else:
        lines[mod_index:mod_index] = color_change_gcode
        gcode_file.writelines(lines)

print('"%s" annotated' % path)