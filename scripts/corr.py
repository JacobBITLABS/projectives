#!/usr/bin/env python
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from boxplot import FIELDS

@click.command()
@click.argument("file", type=click.Path(exists=True))
@click.argument("start", type=int)
@click.argument("end", type=int)
@click.option("--output", default=None, type=click.Path())
@click.option("--format", default="pdf")
@click.option("--gender", default=None, type=int)
@click.option("--prefixes", type=str, default=None)
@click.option("--min-age", type=int, default=None)
@click.option("--max-age", type=int, default=None)
@click.option("--normalize/--no-normalize", is_flag=True, default=False)
@click.option("--baseline/--no-baseline", is_flag=True, default=True)
@click.option("--remove-empty/--no-remove-empty", is_flag=True, default=True)
@click.option("--dpi", type=int, default=300)
def corr(file, start, end, output, format, gender, prefixes, min_age, max_age, normalize, baseline, remove_empty, dpi):
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
    selected_df = df.iloc[:,start:end]
    if remove_empty:
        selected_df = selected_df.dropna(how='all')
    if baseline:
        selected_df = selected_df.div(df['baseline'], axis=0)
    if normalize:
        selected_df = selected_df.div(selected_df.sum(axis=1), axis=0)
    selected_df = selected_df.join(df.loc[selected_df.index, ["age", "gender"]])
    sample_size = selected_df.shape[0]
    selected_columns = selected_df.columns.tolist()
    FIELDS.append("age")
    FIELDS.append("gender")
    # Compute correlation matrix
    corr_matrix = selected_df.corr()

    # Plot correlation matrixs
    fig, ax = plt.subplots(figsize=(12, 9))
    cax = ax.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)

    ax.set_xticks(range(len(selected_columns)))
    ax.set_xticklabels(FIELDS, rotation=45, ha='left')
    ax.set_yticks(range(len(selected_columns)))
    ax.set_yticklabels(FIELDS)

    for (i, j), val in np.ndenumerate(corr_matrix):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')

    plt.title(f'Correlation Matrix (n={sample_size})', pad=20)
    plt.tight_layout()
    if output is None:
        plt.show()
    else:
        plt.savefig(output, format=format, dpi=dpi)
        print(f"Saved to {output}")

if __name__ == "__main__":
    corr()
