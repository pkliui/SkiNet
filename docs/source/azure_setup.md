# Azure

## Azure storage setup

To set up Azure Blob Storage, one needs to complete the following steps:

- Create a valid Azure account and subscription.
- Create a new Resource Group.
- Set up a new Azure Machine Learning Workspace.
- Create a Storage Account to store your images. (Azure automatically creates one with a workspace, but you can use it or create a separate account.)
- Configure authentication and authorization to enable programmatic access to the images.

### Create a Resource group and a workspace

- Login into the Azure portal (given you already have a valid account and a subscription)
- Create a new resource group, under ```Home / Resource groups```, hit ```Create```, invent a name (e.g. pkliui-rg) and select a region for the group (e.g. Central US)
- Next, create a new workspace. In ```All services / Azure Machine Learning```, hit ```Create``` and select ```New workspace``` Specify the resource group, give the workspace a name and  finish with ```Create```. More details about Azure workspaces can be found [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace?view=azureml-api-2&tabs=python).

### Create a Storage account and upload data

Now we need to make a new storage account to keep our images.
- Under ```Home / Storage accounts```, hit ```Create```
- Select the resource group (e.g. pkliui-rg) and specify a globally unique storage account name (e.g. skinetstorage), the region (US Central), performance and redundancy (**LRS for the lowest costs and non-critical scenarios**) and  finish with ```Create```
- Enter the newly created storage account and under ```Storage browser / Blob containers``` select ```Add container``` and give a name to it e.g. skinet-container, click ```Create```
- A newly created container will appear under ```Blob containers```
- Upload the images into the container.

### Create a datastore

- Instead of issuing users with keys for a storage account and interacting with it directly, data are accessed via a datastore. An Azure Machine Learning datastore serves as a reference to an existing Azure storage account and it is a child resource of the respective workspace.
- Datasores can be authenticated either using credentials or via idenity-based method that will be described below.

- You need to link the storage account that holds your data to a new datastore using an account key in order to give AzureML permissions to manage it. In the storage account go to ```Security + Networking / Access keys```  and copy one of the keys.
- In the workspace, hit  ```Studio web URL ``` to open Machine Learning Studio
- Select the workspace and under  ```Assets / Data ``` select  ```Datastores ``` and  ```Create ```
- Enter a name for this new datastore e.g. skinetdatastore
- Select Datastore type e.g. Azure Blob Storage
- Select your newly created storage account e.g. skinetstorage and a respective blob container e.g. skinet-container
- Pick "Save credentials with the datastore or data access", under "Authentication type" select "Account key" and paste the storage account key from the clipboard, click ```Create```.

## Identification of datasets

- The `DatasetKey` Enum is used to uniquely identify datasets throughout the code. The **value** of each enum member is called `dataset_name`.

**Example Enum:**
```python
@unique
class DatasetKey(Enum):
    PH2 = "PH2_DATASET"
    ANOTHERSET = "ANOTHER_SET"
```

- Create a YAML config file (e.g., azure_settings.yaml), where add a section `PATH_ON_DATASTORE` mapping `dataset_name`s from `DatasetKey` to the respective relative paths to data on the datastore.
- The `dataset_name`s in `DatasetKey` Enum must match the YAML keys under `PATH_ON_DATASTORE`. The corresponding YAML value `data_root_on_azure` points to relative paths to data on the datastore.

**Example YAML config**
```yaml
PATH_ON_DATASTORE:
    PH2_DATASET: "PH2DATA/"
    ANOTHER_DATASET: "another/path/"
```

| Enum Member (dataset_key)   | Enum Value or YAML key (dataset_name) | YAML Value (data_root_on_azure)     |
|-----------------------------|---------------------------------------|-------------------------------------|
| PH2                         | "PH2_DATASET"                         | "PH2DATA/"                          |
| ANOTHERSET                  | "ANOTHER_SET"                         | "another/path/"                     |

## Access data in Azure Blob Storage

Data in Azure Blob Storage can be accessed in various ways. In SkiNet, we use AzureMachineLearningFileSystem on local and Blobfuse2 mounts both on local and on Azure hosts.

### Access via AzureMachineLearningFileSystem using Service Principal authentication

- AzureMachineLearningFileSystem is mainly used to access the storage account programmatically during local experimentation (creation of metadata, inspection of images etc.).

- To access AzureMachineLearningFileSystem, we set up a Service Principal. Users are issued with the SP's credentials, including a secret.
- And the SP is granted rights to perform defined actions on particular resources. This approach enables precise control over what each Service Principal can access and manage within Azure.

- More about Service Principals and application registrations can be read [here](https://learn.microsoft.com/en-us/entra/identity-platform/app-objects-and-service-principals?tabs=browser). Please note that using [Managed Identities](https://learn.microsoft.com/en-us/entra/identity/managed-identities-azure-resources/overview) eliminates the need to manage credentials, but this type of authentication is available only for Azure machines.

#### Generate SP credentials

Begin setting up a SP as follows:

- In Azure, go to ```Home / App registrations``` and start a new registration with a name e.g. skinet-sp. Select "Accounts in this organizational directory only" and then "Register"
- Enter the newly created service principal application by clicking on its name in the list of registered applications
- Note "Application (client) ID", "Directory (tenant) ID", you will need them for ```azure_settings.yaml``` file to be able to access data programmatically.
- Under ```Certificates and secrets```, hit ```New client secret``` and create a new token for the service principal. The token will appear only once, so note it well. To keep this token, either create an environment variable, e.g. in python ```os.environ["AZURE_CLIENT_SECRET"] = TOKEN_VALUE ``` or keep it in a separate text file for reading out later.
- In case the same service principal will be used by multiple users, all of them need to have this saved token on their machines.
- Add credentials to the YAML config file (e.g., azure_settings.yaml)

```yaml
AZURE_TENANT_ID: "<tenant-id>"
AZURE_CLIENT_ID: "<client-id>"
SUBSCRIPTION_ID: "<subscription-id>"
RESOURCE_GROUP: "<resource-group>"
WORKSPACE_NAME: "<workspace-name>"
DATASTORE_NAME: "<datastore-name>"
PATH_ON_DATASTORE:
    PH2_DATASET: "PH2DATA/"
    ANOTHER_DATASET: "another/path/"
```

- Now when we have set up the service principal, we need to grant it with appropriate rights

#### Grant right to SP via role assignments

- In order to be able to access blob's data with the SP credentials and AzureMachineLearningFileSystem, we need to grant righs to the SP, which is done via role assignment.
- Go to your Azure Machine Learning Workspace (not studio but the workspace in Azure portal) and under ```Access control (IAM) / Role assignments``` choose "New role assignment"
- Select ```AzureML Data Scientist```, pick the newly created service principal and finish by clicking ```Save```. You should be able to access the data in your data storage account programmatically using the *service principal credentials* and *AzureMachineLearningFileSystem* tools
- Note that in this case an access to Azure Blob Storage is granted at the level of the Workspace and the role enables for data access through AzureMachineLearningFileSystem. For any other way of access, e.g. data mounted via blobfuse, a different role and a different level will be required.

#### Access datasets from code

```python
from SkiNet.Azure.azure_setup import AzureSetup
AzureSetup.service_principal_authentication()

dataset_name= DatasetKey.PH2.value  # "PH2_DATASET"
fs = AzureSetup.get_azureml_filesystem(dataset_name)
```

In this example, dataset_name (DatasetKey.PH2.value == "PH2_DATASET") matches the YAML key PH2_DATASET,
which maps to the path "PH2DATA/" on the Azure datastore. This allows the code to access the correct dataset on Azure.

### Access via Blobfuse2 mounts using managed identity and SP authentication

Azure Blob Storage can be mounted on an Azure machine via Blobfuse2 and accessed without credentials by using an Azure Managed Identity. Blobfuse mounts can also be set up on local machines (in this case other kind of authentication such as SP is required).

#### Create and set up a new Managed Identity (MI)

- To be able to access data whilst on an Azure machine, one should set up managed identity (MI) authentication for that machine.
- In Azure, go to ```Home / Create a resource``` and select "Infrastructure services"
- Under "Idenity" select "Managed Identity"  and in a new windows press "Create"
- Specify the resource group, name and region for the MI.
- Now we want to use this MI to access data in the Azure Blob Storage from within the Azure machine
- Go to the Storage account and under Access Control, select "Add role assignment"
- Select  "Storage Blob Data Contributor" and assign access to the managed identity.

#### Set up Blobfuse

- BlobFuse is an open-source virtual file system driver that integrates Azure Blob Storage with Linux environments. Both local and Azure machines are possible.

- By using BlobFuse, you can mount Azure Storage account containers as a file system, making blob data accessible through standard Linux file operations
- More information about blobfuse2 mounts is [here](https://learn.microsoft.com/en-us/azure/storage/blobs/blobfuse2-what-is)

- You need to st up a BlobFuse config file that contains blob storage account name, container and endpoint. For more info see [here](https://learn.microsoft.com/en-us/azure/storage/blobs/blobfuse2-commands-mount)

- Depending on the mode of access (SPN for service principal and MSI for managed idenity), other fields are required. In example below, the mode of access is set to SP, so the fields for SP credentials are included. In MI mode, the code will ignore these fields and modify the mode accordingly.

**Example YAML config (showing only the storage credentials section)**
```
azstorage:
  type: block # type: block|adls
  account-name: skinetstorage #<name of the storage account>
  container: skinet-container #<name of the storage container to be mounted>
  endpoint: skinetstorage.blob.core.windows.net #<example - https://account-name.blob.core.windows.net>
  mode: spn #key|sas|spn|msi|azcli
  clientid: '${AZURE_CLIENT_ID}'  # AZURE_CLIENT_ID
  tenantid: '${AZURE_TENANT_ID}' # AZURE_TENANT_ID
  clientsecret: '${AZURE_CLIENT_SECRET}'
```


#### Access datasets from code

**Example data mount on Azure machine using MI**

- For training, Azure Blob Storage is expected be mounted on a computational instance with an assigned managed identity
- In this case, data are available to the MI within the scope granted to it and can be accesed from the code without specifying any credentials.

```python
from SkiNet.Utils.project_paths import BLOBFUSE2_CONFIG_PATH  # Default blobfuse2 config path

from SkiNet.Azure.azure_blob_mounter import AzureBlobMounter
mounter = AzureBlobMounter(
    mount_path=path_on_azure_machine_where_to_mount_datas,
    config_path=BLOBFUSE2_CONFIG_PATH,  # Uses default config unless overridden
    is_azure_mount=True
)
mounter.mount()
```

This is wrapped in the "mount_data.py" script which can be callable for Azure mounts as

```python
python mount_data.py --mount-path="SOME_AZURE_MOUNT_PATH" --is-azure-mount --config-path=BLOBFUSE2_CONFIG_PATH
# If you do not specify --config-path, the script will use the default value (BLOBFUSE2_CONFIG_PATH) automatically.
```

**Example data mount on local using SP authentication under the hood**

- It is possible to mount the Blob Storage on a local machine. In this case, a SP auth is required (and has been set up in code).

```python
python mount_data.py --mount-path="SOME_LOCAL_MOUNT_PATH" --config-path=BLOBFUSE2_CONFIG_PATH
# If you do not specify --config-path, the script will use the default value (BLOBFUSE2_CONFIG_PATH) automatically.
```

```python
 python mount_data.py --mount-path /mnt/data --config-path SkiNet/Azure/blobfuse2.yaml
 ```

## Running experiments on Azure

### Create a new Azure compute and set it up using run_on_azure.sh
- Create a new Azure compute machine and provide it with a startup script "run_on_azure.sh" and the name of the Docker image as an argument, e.g. "pkliui/skinet:gpu" assuming that one is publicly available.

- Each time the machine starts, the script sets environment variables, downloads and installs git, docker and blobuse2; also the startup script will pull the latest changes in the repo, mount the Azure Blob Storage as specified in Blobfuse2 config , pull the latest version of the Docker image and run it
-
- Perhaps it will make sense to split this into 2 scripts: install do not need to be repeated every time machine starts.

- Note: Docker should be run with the special options allowing access to the mounted Azure Blob Storage
```
docker run --rm -it\
  --cap-add=SYS_ADMIN \
  --device=/dev/fuse \
  --security-opt apparmor:unconfined \
  --mount "type=bind,src=$HOST_REPO,dst=$CONTAINER_REPO" \
  --mount "type=bind,src=$AZURE_MOUNT_PATH,dst=$CONTAINER_AZURE_MOUNT_PATH" \
  -w "$CONTAINER_REPO" \
  "$IMAGE"
```

- This command bind-mounts the repository from the host to the container and also bind-mounts the Azure Blob Storage (mounted on the host) into the Docker container as specified.


### Run code inside the docker environment as on local