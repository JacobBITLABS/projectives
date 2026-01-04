#!/usr/bin/env python
import click
import matplotlib.pyplot as plt
import pandas as pd

from boxplot import FIELDS

@click.command()
@click.argument("file", type=click.Path(exists=True))
@click.argument("start", type=int)
@click.argument("end", type=int)
@click.option("--stacked", is_flag=True, default=False)
@click.option("--sorted", is_flag=True, default=False)
@click.option("--output", default=None, type=click.Path())
@click.option("--format", default="pdf")
@click.option("--gender", default=None, type=int)
@click.option("--prefixes", type=str, default=None)
@click.option("--min-age", type=int, default=None)
@click.option("--max-age", type=int, default=None)
@click.option("--remove-empty/--no-remove-empty", is_flag=True, default=True)
@click.option("--normalize/--no-normalize", is_flag=True, default=True)
@click.option("--dpi", type=int, default=300)
def barchart(file, start, end, stacked, sorted, output, format, gender, prefixes, min_age, max_age, remove_empty, normalize, dpi):
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
    counts_df = pd.DataFrame({col: df[col].value_counts().reindex(range(1,12+1), fill_value=0)
                            for col in d.columns.tolist()})
    if sorted:
        column2field = {i+1: field for i, field in enumerate(FIELDS)}
        counts_df['total'] = counts_df.sum(axis=1)
        counts_df = counts_df.sort_values('total', ascending=False).drop('total', axis=1)
        fields = [column2field[col] for col in counts_df.index.tolist()]
    else:
        fields = FIELDS
    if normalize:
        counts_df = counts_df.div(counts_df.sum(axis=0)/100, axis=1)
    ax = counts_df.plot(kind='bar', stacked=stacked, figsize=(12, 7))

    plt.title(f'Top-2 and top-3 health priorities for participants with family as top-1 (n={sample_size})', pad=20)
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=45)
    ax.set_xticklabels(fields)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.legend([f"top-{i+1}" for i in range(1, end-start+1)], title='Priorities')
    plt.tight_layout()
    if output is None:
        plt.show()
    else:
        plt.savefig(output, format=format, dpi=dpi)
        print(f"Saved to {output}")

if __name__ == "__main__":
    barchart()
