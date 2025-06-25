import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader

# --- Environment-Agnostic Data Directory Configuration ---
def get_data_dir():
    """
    Determines the correct data directory based on the execution environment.
    This function makes the code portable between Colab, local PyCharm, and cluster.
    """
    # Default path for local/cluster (adjust this for your specific setup if needed)
    # For PyCharm, this assumes 'data' folder is in your project root.
    # For cluster, you'd typically set it to the absolute path where your 'data' folder resides.
    local_or_cluster_project_path = os.getcwd() # Or '/s/project/ml4rg_students/2025/<your_group_folder>/DNA-RNA-Joint-Modeling/' for cluster
    data_directory = os.path.join(local_or_cluster_project_path, 'data/')

    # Attempt to detect Google Colab
    try:
        from google.colab import drive
        # If drive is importable, we are likely in Colab.
        # This will only mount if not already mounted.
        drive.mount('/content/gdrive')
        google_drive_project_path = '/content/gdrive/MyDrive/DnARnAProject/'
        data_directory = os.path.join(google_drive_project_path, 'data/')
        print("Detected Google Colab environment. Using Google Drive path.")
    except ImportError:
        # Not in Colab, use the local/cluster path
        print("Not in Google Colab. Using local/cluster path.")

    # Final check for directory existence (good for all environments)
    if not os.path.isdir(data_directory):
        raise FileNotFoundError(f"Error: The data directory '{data_directory}' does not exist. "
                                "Please ensure your data is located correctly for your environment.")
    return data_directory

# --- Custom PyTorch DataLoader Class ---
class GenomeExpressionDataset(Dataset):
    """
    Custom Dataset for loading DNA sequence and expression data for genomic regions.
    It loads data from pre-processed .npz and .parquet files.
    Handles reverse complement for '-' strand sequences and uses appropriate expression labels.
    """
    def __init__(self, data_dir):
        """
        Initializes the dataset by loading the full sequence and expression arrays
        and the DataFrame of genomic regions.

        Args:
            data_dir (str): The path to the directory containing 'data.npz' and 'regions.parquet'.
        """
        self.data_dir = data_dir

        self.data_npz_path = os.path.join(data_dir, 'data.npz')
        self.regions_parquet_path = os.path.join(data_dir, 'regions.parquet')

        try:
            # Added allow_pickle=True as it can sometimes be necessary for .npz files
            self.data_npz = np.load(self.data_npz_path, allow_pickle=True)
            self.sequence_data = self.data_npz['sequence']
            self.expression_plus_data = self.data_npz['expressed_plus']
            self.expression_minus_data = self.data_npz['expressed_minus']
            self.data_npz.close() # Important to close the loaded npz file handle
        except KeyError as e:
            # Provide more informative error if a key is missing
            available_keys = list(np.load(self.data_npz_path).keys()) if os.path.exists(self.data_npz_path) else "File not found during key check."
            raise RuntimeError(f"KeyError: Key '{e}' not found in {self.data_npz_path}. "
                               f"Available keys: {available_keys}. "
                               "Please check your .npz file structure.")
        except Exception as e:
            raise RuntimeError(f"Could not load data from {self.data_npz_path}. Make sure the file exists and is not corrupted: {e}")

        try:
            self.regions_df = pd.read_parquet(self.regions_parquet_path)
        except Exception as e:
            raise RuntimeError(f"Could not load regions from {self.regions_parquet_path}. Make sure the file exists and is not corrupted: {e}")

        self.num_nucleotides = 5 # A, C, G, T, N (mapped to 0, 1, 2, 3, 4)

        # Define the complement mapping for integer-encoded bases
        # Assuming A=0, C=1, G=2, T=3, N=4
        # Complement: A<->T, C<->G, N<->N
        # 0<->3, 1<->2, 4<->4
        self.complement_map = np.array([3, 2, 1, 0, 4], dtype=np.uint8)


    def __len__(self):
        return len(self.regions_df)

    def _one_hot_encode(self, sequence_segment):
        """
        Converts a sequence segment (array of integer encodings) into a one-hot encoded tensor.
        """
        one_hot_tensor = torch.zeros(len(sequence_segment), self.num_nucleotides, dtype=torch.float32)
        one_hot_tensor.scatter_(1, torch.tensor(sequence_segment).unsqueeze(1).long(), 1)
        return one_hot_tensor

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        region_info = self.regions_df.iloc[idx]

        offset = region_info['offset']
        window_size = region_info['window_size']
        strand = region_info['strand']

        # Extract sequence segment
        sequence_segment = self.sequence_data[offset : offset + window_size].copy() # .copy() to avoid modifying original array

        # Determine the expression label and prepare sequence based on strand
        if strand == '+':
            encoded_sequence = self._one_hot_encode(sequence_segment)
            expression_label = self.expression_plus_data[offset] # Use label at the start of the window from plus strand data
        else: # strand == '-'
            # Reverse complement the sequence
            reverse_complemented_sequence = self.complement_map[sequence_segment][::-1].copy()
            encoded_sequence = self._one_hot_encode(reverse_complemented_sequence)
            expression_label = self.expression_minus_data[offset] # Use label at the start of the window from minus strand data

        expression_label = torch.tensor(expression_label, dtype=torch.long)

        # Return relevant info for the model
        return encoded_sequence, expression_label, region_info.to_dict()

