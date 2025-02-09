import argparse
from pathlib import Path
import dash
from dash import html, dcc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from SkiNet.Plotting.get_data.get_images_and_masks import get_random_sample
from SkiNet.Plotting.adjust_data.adjust_masks import adjust_mask_for_goimage

from SkiNet.ML.utils.configs.dynamic_class_loader import DynamicClassLoader


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

if __name__ == '__main__':

    """
    Example:
    python plot_random_samples.py --dataset-name PH2Dataset --path-to-data /Users/Pavel/Documents/repos_data/UNet/PH2_Dataset_images/PH22 --num-images-to-plot 2
    """

    parser = argparse.ArgumentParser(description="Visualize randomly picked images and masks for a given dataset and number of images")
    parser.add_argument(
        '--dataset-name', 
        type=str, 
        required=True, 
        help="Dataset class name specific to a given data. \
        Available datasets: PH2Dataset"
    )
    parser.add_argument(
        '--path-to-data', 
        type=str, 
        required=True, 
        help="Path to a directory with images and masks."
    )
    parser.add_argument(
        '--num-images-to-plot', 
        type=int, 
        required=False, 
        default=5, 
        help="Number of image-mask pairs to display (default: 5)."
    )    
    args = parser.parse_args()

    # create a new dataset
    loader = DynamicClassLoader(args.dataset_name)
    dataset_class = loader.load_the_class()
    dataset = dataset_class(root_dir=Path(args.path_to_data))

    # initialize a new dash app and specify its layout
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.Div([
            html.Div([
                dcc.Graph(
                    id=f'image-mask-{i}',
                    figure=create_figure_with_subplots(dataset),
                    style={'display': 'inline-block', 'width': '90%', 'padding': '10px'}
                ),
            ], style={'text-align': 'center', 'margin': '20px 0'})
            for i in range(args.num_images_to_plot)  # Ensure this is properly aligned
        ])
    ])

    try:
        app.run_server(debug=True)
    except KeyboardInterrupt:
        print("Server stopped by user.")
