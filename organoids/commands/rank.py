import click
import itertools
import json
import os
import pandas as pd
import shapely
import tqdm

from ..utils import end, start, status

@click.group()
def _rank():
    pass

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

@_rank.command()
@click.argument("directory", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option("--ext", default=".json", help="File extension to search for (default: .json)")
@click.option("--separator", default=",", help="Separator for CSV file (default: ,).")
@click.option("--id-separator", default="-kopi", help="Separator for id in file name (default: -kopi).")
@click.option("--decimal-separator", default=".", help="Decimal separator for CSV file (default: .).")
@click.option("--permutation", default=None, help="Permutation of labels as comma-separated list (default: None).")
@click.option("--permutation-prefix", default="", help="Prefix for permutation labels (default: None). If set, will only permute labels for ids starting with this prefix.")
@click.option("--age-gender", default=None, help="Extra columns for age and gender as json mapping (default: None).")
@click.option("--exclude", type=click.Path(), help="Exclude files as comma-separated list (default: None).")
@click.option("--require-prefix", default=None, help="Require that id starts with this, adding add-prefix if necessary (default: None).")
@click.option("--add-prefix", default=None, help="Add if id does not have required prefix (default: None).")
def rank(directory, output, ext, separator, id_separator, decimal_separator, permutation, permutation_prefix, age_gender, exclude, require_prefix, add_prefix):
    if require_prefix is not None and add_prefix is None:
        add_prefix = require_prefix
    if exclude is not None:
        exclude = exclude.split(",")
    start("Scanning for files")
    todo = [directory]
    found = []
    while todo:
        dir = todo.pop()
        for entry in os.listdir(dir):
            entry_path = os.path.join(dir, entry)
            if os.path.isdir(entry_path):
                todo.append(entry_path)
            elif entry.endswith(ext):
                if exclude is not None and any(ex in entry for ex in exclude):
                    print(f"Excluding {entry}")
                    continue
                found.append(entry_path)
    status(len(found), end='')
    end()
    permutation_counter = 0
    start("Computing rank")
    with open(output, "w") as f:
        header = f"id{separator}{separator.join([f'area{i+1}' for i in range(12)])}{separator}{separator.join([f'rank{i+1}' for i in range(12)])}{separator}{separator.join([f'largest{i+1}' for i in range(12)])}"
        if age_gender is not None:
            age_gender = json.load(open(age_gender, "rt"))
            header += f"{separator}age{separator}gender"
        f.write(f"{header}\n")
        for file in tqdm.tqdm(found):
            data = load_json(file)
            if "shapes" not in data.keys():
                print(f"Warning: {file} has no shapes data")
                continue
            if not id_separator in file:
                print(f"Warning: {file} does not have the expected name format ID{id_separator}.json")
                continue

            # id from file name
            id = file.split("/")[-1].rsplit(id_separator, maxsplit=1)[0]
            if require_prefix and not id.startswith(require_prefix):
                id = f"{add_prefix}{id}"

            poly_data = []
            shapes = data["shapes"]
            for shape in shapes:
                if 'label' not in shape.keys():
                    print(f"Warning: {file} contains shapes without label-key")
                    continue
                label = shape['label'].replace("?", "")
                labels = label.split(",")
                if len(labels) > 1:
                    print(f"Warning: {file} contains invalid label {label}")
                for label in labels:
                    if permutation is not None and id.startswith(permutation_prefix):
                        permutation_counter += 1
                        try:
                            label = permutation.split(",")[int(label)-1]
                        except (ValueError, IndexError):
                            print(f"Warning: {file} contains invalid label {label}")
                            continue
                    try:
                        poly = shapely.geometry.Polygon(shape['points'])
                    except ValueError:
                        print(f"Warning: {file} contains invalid polygon {shape['points']}")
                        continue
                    area = poly.area
                    poly_data.append({'label': label, 'area': area})

            # list areas
            label2area = {int(poly['label']): poly['area'] for poly in poly_data}
            areas = separator.join(str(label2area.get(i+1)).replace('.',decimal_separator) for i in range(12))

            # sort polygons by area in decreasing order
            poly_data.sort(key=lambda x: x['label'], reverse=True)
            new_poly_data = []
            for label, group in itertools.groupby(poly_data, key=lambda x: x['label']):
                _areas = [poly['area'] for poly in group]
                if len(_areas) > 1:
                    print(f"Warning: {file} contains multiple polygons with the same label {label}: {len(_areas)-1} extra labels")
                new_poly_data.append({'label': label, 'area': sum(_areas)/len(_areas)})
            new_poly_data.sort(key=lambda x: x['area'], reverse=True)
            sorted_labels = [poly['label'] for poly in new_poly_data]
            while len(sorted_labels) < 12:
                sorted_labels.append('')
            largest = separator.join(sorted_labels)

            # rank polygons by area
            ranks = separator.join((str(sorted_labels.index(str(i+1))+1) if str(i+1) in sorted_labels else '') for i in range(12))

            # write to CSV file
            line = f'"{id}"{separator}{areas}{separator}{ranks}{separator}{largest}'
            if age_gender is not None:
                meta = age_gender.get(id, {})
                line += f"{separator}{meta.get('age', '')}{separator}{meta.get('gender', 0)}"
            f.write(f"{line}\n")
    end()
    print(f"Permutation counter: {permutation_counter}")
    start("Exporting to Excel")
    df = pd.read_csv(output, sep=separator, decimal=decimal_separator)
    df.to_excel(output.replace(".csv", ".xlsx"), index=False)
    end()
