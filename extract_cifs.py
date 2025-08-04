
import pickle
import gzip
import pandas as pd

def extract_cifs_to_dataframe(input_file, output_file):
    """
    Reads a gzipped pickle file containing CIF data, extracts the data,
    and saves it as a CSV file.

    Args:
        input_file (str): Path to the input gzipped pickle file.
        output_file (str): Path to the output CSV file.
    """
    print(f"Reading data from {input_file}...")
    with gzip.open(input_file, 'rb') as f:
        data = pickle.load(f)

    # Assuming the data is a list of dictionaries, where each dictionary
    # has a 'cif' key containing the CIF string.
    # If the structure is different, this part will need to be adjusted.
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and 'cif' in data[0]:
        cif_list = [item['cif'] for item in data]
        df = pd.DataFrame(cif_list, columns=['cif'])
    elif isinstance(data, list):
        # Fallback if it's just a list of strings
        df = pd.DataFrame(data, columns=['id', 'cif'])
    else:
        print("Unsupported data format in pickle file.")
        # Attempt to create a DataFrame directly, this might not be what the user wants
        # but it's a best guess.
        df = pd.DataFrame(data)


    print(f"Saving DataFrame to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    input_filename = "cifs_v1_train.pkl.gz"
    output_filename = "cifs_v1_train.csv"
    extract_cifs_to_dataframe(input_filename, output_filename)
