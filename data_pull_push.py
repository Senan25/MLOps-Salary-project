from azure.storage.blob import BlobServiceClient
import os
import pandas as pd

def retrieve_data_from_azure():

    connection_string = os.getenv('connection_string')
    container_name = os.getenv('raw_data_container_name')
    blob_name = os.getenv('raw_data_name')

    print(connection_string, container_name, blob_name)

    #try:

    # Initialize the BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    
    # Get the container client
    container_client = blob_service_client.get_container_client(container_name)
    
    # Get the blob client
    blob_client = container_client.get_blob_client(blob_name)
    
    # Download the blob content
    download_file_path = f"./blob-{blob_name}"  # Local path to save the file
    with open(download_file_path, "wb") as file:
        blob_data = blob_client.download_blob()
        file.write(blob_data.readall())
    
    print(f"Blob '{blob_name}' downloaded to '{download_file_path}' successfully.")
    return pd.read_csv(download_file_path)
    # except Exception as e:
    #     print(f"An error occurred: {e}")

