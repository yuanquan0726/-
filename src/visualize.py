import plotly.express as px

from sklearn.datasets import load_digits
from umap import UMAP


def main():
    digits = load_digits()
    umap_2d = UMAP()
    umap_2d.fit(digits.data)

    projections = umap_2d.transform(digits.data)

    fig = px.scatter(
        projections,
        x=0,
        y=1,
        color=digits.target.astype(str),
        labels={"color": "digit"},
    )

    fig.write_html("public/index.html")
