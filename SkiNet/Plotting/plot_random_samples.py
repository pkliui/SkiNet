import numpy as np
import torch
from pathlib import Path

import dash
from dash import html, dcc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from SkiNet.Plotting.get_data.get_images_and_masks import get_random_sample
from SkiNet.Plotting.adjust_data.adjust_masks import adjust_mask_for_goimage

from SkiNet.ML.datasets.ph2_dataset import PH2Dataset


num_images = 4
ph2_dataset = PH2Dataset(
    root_dir=Path("/Users/Pavel/Documents/repos_data/UNet/PH2_Dataset_images/PH22")
)

def create_figure_with_subplots(data_set):
    """
    Create a figure with subplots: one for the image and one for the mask
    """
    sample = get_random_sample(data_set)
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Image", "Mask"))
    fig.add_trace(go.Image(z=sample['image']), row=1, col=1)
    fig.add_trace(go.Image(z=adjust_mask_for_goimage(sample['mask'])), row=1, col=2)

    fig.update_layout(
        title=sample['name'],
        xaxis={'showgrid': False, 'visible': False},
        yaxis={'showgrid': False, 'visible': False},
        margin=dict(l=10, r=10, t=30, b=10)
    )
    return fig

app = dash.Dash(__name__)
app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Graph(
                id=f'image-mask-{i}',
                figure=create_figure_with_subplots(ph2_dataset),
                style={'display': 'inline-block', 'width': '90%', 'padding': '10px'}
            ),
        ], style={'text-align': 'center', 'margin': '20px 0'})
        for i in range(num_images)
    ])
])

if __name__ == '__main__':
    try:
        app.run_server(debug=True)
    except KeyboardInterrupt:
        print("Server stopped by user.")
