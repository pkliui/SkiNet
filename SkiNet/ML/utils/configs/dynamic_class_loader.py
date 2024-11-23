import importlib
import inspect
import logging
from typing import Type, Optional


class DynamicClassLoader:
    """
    Dynamically loads a class given its name
    """

    def __init__(self, class_name: str):
        """
        :param class_name: Name of a class
        """
        self.class_name = class_name 
        self.module_name = self._get_module_name(self.class_name)
        """Module name for a specified class"""

    def _get_module_name(self, class_name: str) -> str:
        """
        Return the module's name for a specified class

        In the future this could be changed to find the location automatically
        """
        class_names = ["PH2Dataset"]
        if class_name in class_names :
            if class_name == "PH2Dataset":
                return "SkiNet.ML.datasets.ph2_dataset"
        else:
            raise ValueError("The specified class not available for dynamic downloading yet. \
                             Available classes are {class_names}")

    def load(self) -> Optional[object]:
        """
        :return: The specified class if found, otherwise None.
        """
        try:
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
            logging.debug(f"Error loading module {self.module_name}: {str(e)}")
            if str(e) != "":
                logging.warning(f"Error loading module {self.module_name}: {str(e)}")
            return None
