# Azure setup

## Set up data storage

In order to keep the images in and access them from Azure Blob Storage, we need to do the following. 

- Set up a valid Azure account and a subscription
- Set up a new Resource group
- Set up a new Azure Machine Learning Workspace
- Set up a new Storage account to keep the images. By default, Azure creates a storage account upon creating a workspace. You could re-use it or create a separate one.
- Set up an authentication and authorization to be able to access the images programmatically

### Create a Resource group and a workspace

- Login into the Azure portal (given you already have a valid account and a subscription)
- Create a new resource group, under ```Home / Resource groups```, hit ```Create```, invent a name (e.g. pkliui-rg) and select a region for the group (e.g. Central US)
- Next, create a new workspace. In ```All services / Azure Machine Learning```, hit ```Create``` and select ```New workspace``` Specify the resource group, give the workspace a name and  finish with ```Create```. More details about Azure workspaces can be found [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace?view=azureml-api-2&tabs=python). 

### Create a Storage account and upload data

Now we need to make a new storage account to keep our images. 
- Under ```Home / Storage accounts```, hit ```Create```
- Select the resource group (e.g. pkliui-rg) and specify a globally unique storage account name (e.g. skinetstorage), the region (US Central), performance and redundancy (LRS or the lowest costs and non-critical scenarios) and  finish with ```Create```
- Enter the newly created storage account and under ```Storage browser / Blob containers``` select ```Add container``` and give a name to it e.g. skinet-container, click ```Create```
- A newly created container will appear under ```Blob containers```. Upload the images tinto the container.

### Create a datastore

- You need to link the storage account that holds your data to a new datastore using an account key in order to give AzureML permissions to manage it. In the storage account go to ```Security + Networking / Access keys```  and copy one of the keys.
- In the workspace, hit  ```Studio web URL ``` to open Machine Learning Studio
- Select the workspace and under  ```Assets / Data ``` select  ```Datastores ``` and  ```Create ``` 
- Enter a name for this new datastore e.g. skinetdatastore
- Select Datastore type e.g. Azure Blob Storage
- Select your newly created storage account e.g. skinetstorage and a respective blob container e.g. skinet-container
- Pick "Save credentials with the datastore or data access", under "Authentication type" select "Account key" and paste the storage account key from the clipboard, click ```Create```.
  
### Create a Service Principal

- To be able to access the storage account programmatically, we will use a Service Principal. More about Service Principals and application registrations can be read [here](https://learn.microsoft.com/en-us/entra/identity-platform/app-objects-and-service-principals?tabs=browser). Please note that using [Managed Identities](https://learn.microsoft.com/en-us/entra/identity/managed-identities-azure-resources/overview) eliminates the need to manage credentials, but this option is apparently available only for Azure machines. At the moment, I am using a local machine.

- In short, Service Principals are based on assigning roles to different identities by granting them rights for specific actions to access specific resources. This allows for a precise control over the scope of the access. So instead of issuing users with the key for the storage account, granting them unlimited rights, we grant the keys to the datastore. And users are being issued with a Service Principal credential that allows access only within a limited scope, defined by the roles permissions.

- In Azure, go to ```Home / App registrations``` and start a new registration with a name e.g. skinet-sp. Select "Accounts in this organizational directory only" and then "Register"
- Enter the newly created service principal application by clicking on its name in the list of registered applications
- Note "Application (client) ID", "Directory (tenant) ID", you will need them for ```azure_settings.yaml``` file to be able to access data programmatically.
- Under ```Certificates and secrets```, hit ```New client secret``` and create a new token for the service principal. The token will appear only once, so note it well. To keep this token, either create an environment variable, e.g. in python ```os.environ["AZURE_CLIENT_SECRET"] = TOKEN_VALUE ``` or keep it in a separate text file for reading out later.
- In case the same service principal will be used by multiple users, all of them need to have this saved token on their machines.

### Grant rights to the service principal

- Now when we have set up the service principal, we need to grant it with appropriate rights
- Go to your Azure Machine Learning Workspace (not studio but the workspace in Azure portal) and under ```Access control (IAM) / Role assignments``` choose "New role assignment"
- Select ```AzureML Data Scientist```, pick the newly created service principal and finish by clicking ```Save```. You should be able to access the data in your data storage account programmatically using the service principal and AzureMachineLearningFileSystem tools

### Authentication from the code
To access data stored in the blob container from within the code, one must do an authentication with the SP created above. This is done by calling [```service_principal_authentication()```](https://github.com/pkliui/SkiNet/blob/main/SkiNet/Azure/azure_setup.py) method