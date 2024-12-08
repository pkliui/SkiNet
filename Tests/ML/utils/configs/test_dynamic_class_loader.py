"""
Test DynamicClassLoader class
"""

import sys
from unittest.mock import patch, MagicMock
from SkiNet.ML.utils.configs.dynamic_class_loader import DynamicClassLoader



def test_load_successful_default_mapping():
    """
    Test successful dynamic loading of a class by its name using DynamicClassLoader and default class-module mapping
    """

    # Set the name of the class to load and the module where it is located
    class_name = "MockedDataset"
    module_name = "mocked.module"


    # Mock the module in sys.modules
    with patch("SkiNet.ML.utils.configs.dynamic_class_loader.default_class_module_mapping") as mock_default_mapping, \
         patch.dict("sys.modules", {module_name: MagicMock()}):

        mapping_dict = {class_name: module_name}
        mock_default_mapping.return_value = mapping_dict

        # Create a new class with the name "class_name"
        class MockedDataset:
            def __len__(self):
                return 0

        # Explicitly set the __module__ attribute mimicking the location of the class in that module (for inspect.getmodule to work)
        MockedDataset.__module__ = module_name

        # Get the mocked module from sys.modules and add the class to that mocked module
        mocked_module = sys.modules[module_name]
        setattr(mocked_module, class_name, MockedDataset)

        # Inject the mocked mapping into the DynamicClassLoader
        loader = DynamicClassLoader(class_name)
        loaded_class = loader.load_the_class()

        assert loaded_class == MockedDataset


def test_load_successful_custom_mapping():
    """
    Test successful dynamic loading of a class by its name using DynamicClassLoader with a custom class-to-module mapping
    """

    # Define the custom mapping for the test
    class_name = "CustomMockedDataset"
    module_name = "custom.mocked.module"
    custom_mapping = {class_name: module_name}

    # Mock the module in sys.modules
    with patch.dict("sys.modules", {module_name: MagicMock()}):
        
        # Create a new class with the name "class_name"
        class CustomMockedDataset:
            def __len__(self):
                return 0

        # Explicitly set the __module__ attribute mimicking the location of the class in that module (for inspect.getmodule to work)
        CustomMockedDataset.__module__ = module_name

        # Get the mocked module from sys.modules and add the class to that mocked module
        mocked_module = sys.modules[module_name]
        setattr(mocked_module, class_name, CustomMockedDataset)

        # Initialize DynamicClassLoader with the custom mapping
        loader = DynamicClassLoader(class_name, class_to_module_mapping=custom_mapping)
        loaded_class = loader.load_the_class()

        assert loaded_class == CustomMockedDataset