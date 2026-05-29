Datasets API Reference
======================

Dataset Factories
-----------------

.. autofunction:: SkiNet.ML.datasets.dataset_factory.create_segmentation_datasets_from_config

.. autoclass:: SkiNet.ML.datasets.dataset_factory.DatasetSplit
   :members:
   :no-index:

.. autoclass:: SkiNet.ML.datasets.dataset_factory.DatasetFactory
   :members:

.. autoclass:: SkiNet.ML.datasets.dataset_factory.SegmentationDatasetFactory
   :members:

----

Datasets
--------

.. autoclass:: SkiNet.ML.datasets.segmentation_dataset.BaseDataset
   :members:

.. autoclass:: SkiNet.ML.datasets.segmentation_dataset.SegmentationDataset
   :members:

----

Supported Experiment Types
--------------------------

.. list-table::
   :header-rows: 1

   * - ``ExperimentType``
     - Factory
     - Dataset class
   * - ``SEGMENTATION``
     - :py:class:`~SkiNet.ML.datasets.dataset_factory.SegmentationDatasetFactory`
     - :py:class:`~SkiNet.ML.datasets.segmentation_dataset.SegmentationDataset`

----

Extending
---------

To support a new experiment type, subclass
:py:class:`~SkiNet.ML.datasets.dataset_factory.DatasetFactory` and register it:

.. code-block:: python

   class ClassificationDatasetFactory(DatasetFactory):
       def create_datasets(self, config: ExperimentConfig) -> DatasetSplit:
           ...

   dataset_factories = {
       ExperimentType.SEGMENTATION:   SegmentationDatasetFactory(),
       ExperimentType.CLASSIFICATION: ClassificationDatasetFactory(),
   }

----

Internals
---------

:py:meth:`~SkiNet.ML.datasets.dataset_factory.SegmentationDatasetFactory.create_datasets`
runs three steps in order:

1. **Split** — :py:func:`~SkiNet.Utils.data.split_data.split_segmentation_metadata`
   partitions the metadata DataFrame into train/val/test subsets.
2. **Transform** — :py:func:`~SkiNet.ML.transformations.transform_data.get_transform_from_config`
   builds mode-specific augmentation pipelines.
3. **Construct** — one :py:class:`~SkiNet.ML.datasets.segmentation_dataset.SegmentationDataset`
   is instantiated per split, each receiving its corresponding dataframe,
   transform branch, and :py:class:`~SkiNet.ML.utils.model_utils.MLWorkflowState` mode.
