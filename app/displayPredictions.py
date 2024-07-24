import json
from google.cloud import storage
import os

def download_blob(bucket_name, source_blob_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    return json.loads(blob.download_as_string())

def save_to_bucket(bucket_name, destination_blob_name, data):
    """Save the combined data to Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(data)

def combine_predictions(true_labels_path, predictions_path):
    # Download data
    true_labels = download_blob('predict-breed-models', true_labels_path)
    predictions = download_blob('predict-breed-models', predictions_path)

    # Combine data
    combined_data = [
        {"image_id": i + 1, "true_label": t, "prediction": p}
        for i, (t, p) in enumerate(zip(true_labels, predictions))
    ]

    # Convert combined data to JSON string
    combined_json = json.dumps({"results": combined_data}, indent=4)

    # Save the combined JSON to a new file in the bucket
    save_to_bucket('predict-breed-models', 'combined_predictions.json', combined_json)

    print("Combined data saved successfully.")

# Example usage
combine_predictions('true_labels_20240403-204919.json', 'predictions_20240403-204919.json')


def create_breed_index_mapping(directory_path):
    # List all directories (breeds) and sort them alphabetically
    breeds = sorted(os.listdir(directory_path))

    # Create a dictionary mapping from breed to index
    breed_to_index = {breed: idx for idx, breed in enumerate(breeds)}

    # Optionally, save this mapping to a JSON file for later use
    import json
    with open('breed_index_mapping.json', 'w') as file:
        json.dump(breed_to_index, file)

    return breed_to_index


def upload_dict_to_bucket(bucket_name, destination_blob_name, dictionary):
    """Upload a dictionary as a JSON file to Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Convert dictionary to JSON string
    json_data = json.dumps(dictionary, indent=4)
    blob.upload_from_string(json_data, content_type='application/json')
    print(f"Dictionary uploaded successfully to {destination_blob_name}")

# Replace the path below with the actual path to your CroppedImages\train directory
directory_path = r'C:\Users\robby\OneDrive\Desktop\SmartBreed\SmartDogBreed\DogBreed\Images\images\CroppedImages\test'
breed_to_index = create_breed_index_mapping(directory_path)
upload_dict_to_bucket('predict-breed-models', 'breed_to_index.json', breed_to_index)

def download_json_as_dict(bucket_name, source_blob_name):
    """Downloads a JSON file and converts it to a dictionary."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    return json.loads(blob.download_as_string())


def load_json_from_gcs(bucket_name, file_name):
    """ Load JSON data from a Google Cloud Storage bucket """
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    return json.loads(blob.download_as_string())

def combine_predictions_with_breeds(true_labels_path, predictions_path, breed_index_path):
    # Download data
    true_labels = download_json_as_dict('predict-breed-models', true_labels_path)
    predictions = download_json_as_dict('predict-breed-models', predictions_path)
    breed_to_index = download_json_as_dict('predict-breed-models', breed_index_path)

    # Reverse the breed_to_index to get index_to_breed mapping
    index_to_breed = {v: k for k, v in breed_to_index.items()}

    # Combine data
    combined_data = [
        {
            "image_id": i + 1,
            "true_label": index_to_breed.get(t),
            "prediction": index_to_breed.get(p),
            "true_label_index": t,
            "prediction_index": p
        }
        for i, (t, p) in enumerate(zip(true_labels, predictions))
    ]

    # Convert combined data to JSON string
    combined_json = json.dumps({"results": combined_data}, indent=4)

    # Optionally save the combined JSON to a new file in the bucket or return directly
    return combined_json

# Example usage
result_json = combine_predictions_with_breeds('true_labels_20240403-204919.json', 'predictions_20240403-204919.json', 'breed_to_index.json')
print(result_json)