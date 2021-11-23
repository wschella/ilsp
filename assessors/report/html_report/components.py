from matplotlib.figure import Figure
import pandas as pd
import matplotlib.pyplot as plt

from assessors.report.html_report.renderer import Component


class Plot(Component):
    def __init__(self, fig: Figure) -> None:
        self.fig = fig

    def render(self) -> str:
        plt.close(fig=self.fig)
        return f'<img src="data:image/png;base64,{fig_to_base64(self.fig)}">'


def fig_to_base64(fig):
    import base64
    from io import BytesIO

    buf = BytesIO()
    fig.savefig(buf, format='png')
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return encoded


class Quantiles(Component):
    """
    https://pandas.pydata.org/docs/reference/api/pandas.arrays.IntervalArray.html
    https://pandas.pydata.org/docs/reference/api/pandas.IntervalIndex.html
    """

    def __init__(self, series: pd.Series, n_quantiles: int = 4) -> None:
        self.series = series
        self.n_quantiles = n_quantiles

    def render(self) -> str:
        from pandas.arrays import IntervalArray

        quant_size = len(self.series) / self.n_quantiles
        quants = pd.qcut(self.series, self.n_quantiles)
        ia: IntervalArray = quants.dtype.categories

        breakpoints = list(ia.right.values)  # type: ignore
        li = "\n".join(f'<li>{li}</li>' for li in breakpoints)
        return f'''
        <div>
            <p>
                <em>{self.n_quantiles}</em> quantiles,
                <em>{quant_size}</em> per quantile,
                <em>{len(self.series)}</em> total
            </p>
            <p>
                median: {self.series.median():.2f},
                mean:   {self.series.mean():.2f},
            </p>
            <ul>
                {li}
            </ul>
        </div>
        '''
