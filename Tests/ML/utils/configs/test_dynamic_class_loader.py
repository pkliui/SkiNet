import sys
import pytest
from unittest.mock import patch, MagicMock
from SkiNet.ML.utils.configs.dynamic_class_loader import DynamicClassLoader
from torch.utils.data import Dataset


# set the names of the module where we expect the class to be and the class name itself
module_name = "module.with.classtoload"
class_name = "ClassToLoad"

# create a new class with the name "class_name"
class ClassToLoad:
    def __len__(self):
        return 0
    

def test_load_successful():
    """
    Test successful dynamic loading the class by its name 
    and module's location using DynamicClassLoader
    """

    # Explicitly set the __module__ attribute mimicking the location of the class in that module
    ClassToLoad.__module__ = module_name

    # mock the module and create an actual class with the name "class_name" in it
    with patch.dict("sys.modules", {module_name: MagicMock()}):
        mocked_module = sys.modules[module_name]
        setattr(mocked_module, class_name, ClassToLoad)

        # Test loading the class
        loader = DynamicClassLoader(module_name, class_name)
        loaded_class = loader.load()
        assert loaded_class == ClassToLoad


def test_load_nonexistent_module():
    """
    Test loading the specified class in a non-existent module
    """
    loader = DynamicClassLoader("nonexistent_module", class_name)
    loaded_class = loader.load()
    assert loaded_class is None


def test_load_class_not_in_module():
    """
    Test when the class is not actually in the module
    """

    class ClassNotInModule:
        pass

    # mock the module but do not create any class  in it
    with patch.dict("sys.modules", {module_name: MagicMock()}):
        mocked_module = sys.modules[module_name]

        # Test loading the class
        loader = DynamicClassLoader(module_name, "ClassNotInModule")
        loaded_class = loader.load()
        assert loaded_class is None
