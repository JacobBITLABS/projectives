#!/usr/bin/env python
import click
import matplotlib.pyplot as plt
import pandas as pd

FIELDS = ["Friends", "Family", "Disease history", "Healthcare specialists", "Government", "Health information", "Educational background", "Foreign languages", "Digital technologies", "Money for services", "Money for healthy food", "Stress handling ability"]

def clean(s):
    while s[0].isalpha():
        s = s[1:]
    return int(s)

@click.command()
@click.argument("file", type=click.Path(exists=True))
@click.argument("start", type=int)
@click.argument("end", type=int)
@click.option("--invert", is_flag=True, default=False)
@click.option("--top-limit", type=float)
@click.option("--sorted", is_flag=True, default=False)
@click.option("--output", default=None, type=click.Path())
@click.option("--format", default="pdf")
@click.option("--gender", default=None, type=int)
@click.option("--prefixes", type=str, default=None)
@click.option("--min-age", type=int, default=None)
@click.option("--max-age", type=int, default=None)
@click.option("--remove-empty/--no-remove-empty", is_flag=True, default=True)
def boxplot(file, start, end, invert, top_limit, sorted, output, format, gender, prefixes, min_age, max_age, remove_empty):
    df = pd.read_excel(file)
    if gender is not None:
        df = df[df['gender'] == gender]
    if min_age is not None:
        df = df[df['age'] >= min_age]
    if max_age is not None:
        df = df[df['age'] <= max_age]
    if prefixes is not None:
        prefixes = prefixes.split(",")
        mask = df["id"].str.startswith(tuple(prefixes))
        df = df.loc[mask]
    d = df.iloc[:,start:end]
    if remove_empty:
        d = d.dropna(how='all')
    sample_size = d.shape[0]
    if sorted:
        column2field = {i+1: field for i, field in enumerate(FIELDS)}
        medians = d.median().sort_values(ascending=True)        
        d = d[medians.index]
        fields = [column2field[clean(col)] for col in medians.index.tolist()]
    else:
        fields = FIELDS
    plt.figure(figsize=(12,6))
    d.boxplot()
    if invert:
        plt.gca().invert_yaxis()
    plt.xticks(range(1, 12+1), fields, rotation=45)
    if top_limit:
        plt.ylim(top=top_limit)
    plt.title(f'Boxplot for priorities (n={sample_size})', pad=20)
    plt.tight_layout()
    if output is None:
        plt.show()
    else:
        plt.savefig(output, format=format)
        print(f"Saved to {output}")

if __name__ == "__main__":
    boxplot()
