"""
Plot an arbitrary number of images and masks in a dataset side by side, given either its name or provided as a dataset obejct
Can be used from the command line or from code
"""
import argparse
import logging
from pathlib import Path
from typing import Union
import dash
from dash import html, dcc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch.utils.data.dataset import Dataset

from SkiNet.Plotting.get_data.get_images_and_masks import get_random_sample
from SkiNet.Plotting.adjust_data.adjust_masks import adjust_mask_for_goimage
from SkiNet.ML.utils.configs.dynamic_class_loader import DynamicClassLoader

import logging
from SkiNet.Utils.loggers import stdout_logging


def create_images_masks_subplot(data_set, sample_index_to_plot, random_sample=True):
    """
    Create a figure where images and masks are placed side by side as subplots

    :param data_set: A dataset object, e.g. PH2Dataset returning one image and one mask at a time by providing the sample number
    :param sample_num_to_plot: Number of a sample to plot
    :param random_sample: If True, a random sample is picked from the dataset, default is True

    :return fig: A figure with two subplots, one for an image and one for a mask
    """
    # get a random sample - in this case sample_index_to_plot is not used
    if random_sample:
        sample = get_random_sample(data_set)
        sample_name = sample['name']
    else:
        # or get a specific sample
        sample = data_set[sample_index_to_plot]
        sample_name = Path(data_set.images_list[sample_index_to_plot]).parent.parent.name
    #
    # make a figure
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Image", "Mask"))
    fig.add_trace(go.Image(z=sample['image']), row=1, col=1)
    fig.add_trace(go.Image(z=adjust_mask_for_goimage(sample['mask'])), row=1, col=2)
    fig.update_layout(
        title=sample_name,
        xaxis={'showgrid': False, 'visible': False},
        yaxis={'showgrid': False, 'visible': False},
        margin=dict(l=10, r=10, t=30, b=10)
    )
    return fig

def plot_images_masks_side_by_side(dataset_name: str = None, dataset: Dataset = None, num_images_to_plot: int = 1, random_sample: bool = True, path_to_data: Union[str, Path] = None):
    """
    Plot an arbitrary number of images and masks in a dataset, given either its name or provided as a dataset obejct

    :param dataset_name: Name of a dataset class specific to a given data specified in path_to_data. If dataset_name is provided, dataset is not needed. 
    :param dataset: A dataset object yielding a dictionary of an image and a mask. If provided, dataset_name is not needed. Can be used only if run from code.
    :param num_images_to_plot: Number of image-mask pairs to display
    :param random_sample: If True, a random sample is picked from the dataset, default is True
    :param path_to_data: Path to a directory with images and masks or str
    """
    #
    # load the dataset by providing its name (respective dataset calss should exist as checked in the DynamicClassLoader)
    if dataset_name:
        if not path_to_data:
            raise ValueError("path_to_data must be provided if dataset_name is used")
        loader = DynamicClassLoader(dataset_name)
        dataset_class = loader.load_the_class()
        dataset = dataset_class(data_root=Path(path_to_data))
    # if a dataset object is provided, use it, path_to_data is not needed
    elif dataset:
        dataset_name = dataset.__class__.__name__
        if path_to_data:
            logging.getLogger().info("path_to_data is not used as a dataset object has been provided")
    elif not dataset and not dataset_name:
        raise ValueError("Either dataset or dataset_name must be provided")
    elif dataset_name and dataset:
        raise ValueError("Provide either dataset_name or dataset, not both")

    # start dash application
    # run for a given number of images
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.Div([
            html.Div([
                dcc.Graph(
                    id=f'image-mask-{sample_index_to_plot}',
                    figure=create_images_masks_subplot(dataset, sample_index_to_plot, random_sample=random_sample),
                    style={'display': 'inline-block', 'width': '90%', 'padding': '10px'}
                ),
            ], style={'text-align': 'center', 'margin': '20px 0'})
            for sample_index_to_plot in range(num_images_to_plot)
        ])
    ])

    app.run_server(debug=True)

if __name__ == '__main__':
    stdout_logging(logging.DEBUG)

    parser = argparse.ArgumentParser(description="Plot an arbitrary number of images and masks in a dataset, given either its name or provided as a dataset obejct")
    parser.add_argument('--dataset-name', type=str, required=True, help="Name of a dataset class specific to a given data specified in path_to_data. If dataset_name is provided, dataset is not needed. ")
    parser.add_argument('--dataset', type=Dataset, help="A dataset object yielding a dictionary of an image and a mask. If provided, dataset_name is not needed")
    parser.add_argument('--num-images-to-plot', type=int, default=1, help="Number of image-mask pairs to display.")
    parser.add_argument('--random-sample', action='store_true', help="If True, a random sample is picked from the dataset, default is True")
    parser.add_argument('--path-to-data', type=str, help="Path to a directory with images and masks.")
    args = parser.parse_args()

    if args.dataset_name:
        plot_images_masks_side_by_side(dataset_name=args.dataset_name, num_images_to_plot=args.num_images_to_plot, random_sample=args.random_sample, path_to_data=args.path_to_data)
    # allow using the dataset object directly only from code
    elif args.dataset:
        raise ValueError("Provide dataset_name and path_to_data if calling from command line")