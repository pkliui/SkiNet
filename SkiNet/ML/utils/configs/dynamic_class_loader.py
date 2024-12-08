import importlib
import inspect
import logging
from typing import Optional
from SkiNet.ML.utils.configs.default.class_module_mapping import default_class_module_mapping


class DynamicClassLoader:
    """
    Dynamically loads a class by its name using a default dictionary mapping class names to their respective locations
    """

    def __init__(self, class_name: str, class_to_module_mapping: Optional[dict] = None):
        """
        :param class_name: Name of a class to load dynamically
        :param class_to_module_mapping: Dictionary mapping class names to their respective locations in modules, 
            e.g. {"class_name": "module.where.this.class.is"}
        """
        self.class_name = class_name 
        self.class_to_module_mapping =  class_to_module_mapping or default_class_module_mapping()

    def load_the_class(self) -> Optional[object]:
        """
        Load the specified class dynamically.
        
        :return: The specified class if found, otherwise None.
        """
        try:
            self.module_name = self.class_to_module_mapping[self.class_name]
            """Module name for the specified class"""

            # first import the specified module
            imported_module = importlib.import_module(self.module_name)
            logging.debug(f"Importing {imported_module}")

            # get the specified class from that module
            loaded_class = next(
                found_class for found_name, found_class in inspect.getmembers(imported_module)
                # check if  "found_class" is a class and its name is that of the provided "class_name"
                # and additionally check if the found_class is actually being defined within the specified imported_module
                if inspect.isclass(found_class) and found_name == self.class_name 
                and inspect.getmodule(found_class) == imported_module
            )
            logging.debug(f"Found class {loaded_class} in file {imported_module}")
            return loaded_class
        except Exception as e:
            logging.debug(f"Error loading module  for class {self.class_name}: {str(e)}")
            if str(e) != "":
                logging.warning(f"Error loading module for class  {self.class_name}: {str(e)}")
            return None
