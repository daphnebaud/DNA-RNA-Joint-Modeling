# data_loader.py

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
    local_or_cluster_project_path = os.getcwd()
    data_directory = os.path.join(local_or_cluster_project_path, 'data/')

    # Attempt to detect Google Colab
    try:
        from google.colab import drive
        # If drive is importable, we are likely in Colab.
        # This will only mount if not already mounted.
        # drive.mount('/content/gdrive') # Uncomment if you need to mount every time
        google_drive_project_path = '/content/gdrive/MyDrive/DnARnAProject/' # Adjust if your project is elsewhere
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
    It loads data from pre-processed .npz and .parquet files, and also genomic
    annotation from a .gff3 file.
    Handles reverse complement for '-' strand sequences and uses appropriate expression labels.
    """
    def __init__(self, data_dir):
        """
        Initializes the dataset by loading the full sequence and expression arrays,
        the DataFrame of genomic regions, and the GFF3 annotation file.

        Args:
            data_dir (str): The path to the directory containing 'data.npz',
                            'regions.parquet', and 'ensemble_annotation.gff3'.
        """
        self.data_dir = data_dir

        self.data_npz_path = os.path.join(data_dir, 'data.npz')
        self.regions_parquet_path = os.path.join(data_dir, 'regions.parquet')
        self.gff3_path = os.path.join(data_dir, 'ensemble_annotation.gff3') # New GFF3 path

        # --- Load data.npz ---
        try:
            self.data_npz = np.load(self.data_npz_path, allow_pickle=True)
            self.sequence_data = self.data_npz['sequence']
            self.expression_plus_data = self.data_npz['expressed_plus']
            self.expression_minus_data = self.data_npz['expressed_minus']
            self.data_npz.close() # Important to close the loaded npz file handle
            print(f"Successfully loaded {self.data_npz_path}")
        except KeyError as e:
            available_keys = list(np.load(self.data_npz_path).keys()) if os.path.exists(self.data_npz_path) else "File not found during key check."
            raise RuntimeError(f"KeyError: Key '{e}' not found in {self.data_npz_path}. "
                               f"Available keys: {available_keys}. "
                               "Please check your .npz file structure.")
        except Exception as e:
            raise RuntimeError(f"Could not load data from {self.data_npz_path}. Make sure the file exists and is not corrupted: {e}")

        # --- Load regions.parquet ---
        try:
            self.regions_df = pd.read_parquet(self.regions_parquet_path)
            print(f"Successfully loaded {self.regions_parquet_path}")
        except Exception as e:
            raise RuntimeError(f"Could not load regions from {self.regions_parquet_path}. Make sure the file exists and is not corrupted: {e}")

        # --- Load ensemble_annotation.gff3 ---
        try:
            # GFF3 files are tab-separated and have 9 columns by convention
            # Comments start with # and should be skipped.
            gff_columns = ['seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes']
            self.gff_df = pd.read_csv(
                self.gff3_path,
                sep='\t',
                comment='#', # Skip lines starting with #
                header=None,
                names=gff_columns,
                dtype={'start': int, 'end': int, 'score': str, 'phase': str}, # Score/phase can be '.' for missing
                on_bad_lines='warn' # Warn if lines can't be parsed
            )
            print(f"Successfully loaded {self.gff3_path}. Number of annotations: {len(self.gff_df)}")
            # Optional: Further parse the 'attributes' column into a dictionary or separate columns
            # This can be done here or on-the-fly in __getitem__ if needed.
            # For general exploration, just loading it is enough for now.
        except FileNotFoundError:
            print(f"Warning: {self.gff3_path} not found. Proceeding without GFF3 annotations.")
            self.gff_df = pd.DataFrame(columns=gff_columns) # Create an empty DataFrame
        except Exception as e:
            raise RuntimeError(f"Could not load GFF3 annotations from {self.gff3_path}. Make sure the file exists and is not corrupted: {e}")


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
        # Using .long() for scatter_ as it expects LongTensor indices
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
        # .copy() is used to ensure memory safety if sequence_data is memory-mapped
        sequence_segment = self.sequence_data[offset : offset + window_size].copy()

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

        # You could potentially retrieve relevant GFF3 annotations here
        # based on region_info (e.g., chromosome, start, end).
        # For simplicity, this example just returns the standard data.
        # If needed, you'd add logic like:
        # relevant_annotations = self.gff_df[
        #     (self.gff_df['seqid'] == region_info['chromosome']) &
        #     (self.gff_df['start'] < region_info['end']) &
        #     (self.gff_df['end'] > region_info['start'])
        # ]
        # And then include `relevant_annotations.to_dict('records')` in the return tuple.

        # Return relevant info for the model
        return encoded_sequence, expression_label, region_info.to_dict()

# --- Example Usage (Run this block if the script is executed directly) ---
if __name__ == "__main__":
    print("--- Running data_loader.py Example ---")

    # 1. Get the data directory
    try:
        data_dir = get_data_dir()
        print(f"Resolved data directory: {data_dir}")
    except FileNotFoundError as e:
        print(e)
        print("Please ensure your 'data' folder is set up correctly with 'data.npz', 'regions.parquet', and 'ensemble_annotation.gff3'.")
        exit() # Stop execution if data directory is not found

    # Dummy data creation for testing purposes if files don't exist
    # In a real scenario, you would have these files pre-generated.
    if not os.path.exists(os.path.join(data_dir, 'data.npz')):
        print("Creating dummy data.npz for demonstration...")
        dummy_sequence = np.random.randint(0, 5, size=10000, dtype=np.uint8)
        dummy_expressed_plus = np.random.randint(0, 2, size=10000, dtype=np.int64) # Binary expression
        dummy_expressed_minus = np.random.randint(0, 2, size=10000, dtype=np.int64)
        np.savez(os.path.join(data_dir, 'data.npz'),
                 sequence=dummy_sequence,
                 expressed_plus=dummy_expressed_plus,
                 expressed_minus=dummy_expressed_minus)

    if not os.path.exists(os.path.join(data_dir, 'regions.parquet')):
        print("Creating dummy regions.parquet for demonstration...")
        dummy_regions_df = pd.DataFrame({
            'offset': np.arange(0, 9000, 100), # Start of windows
            'window_size': 100, # Fixed window size for simplicity
            'strand': np.random.choice(['+', '-'], size=90),
            'chromosome': 'chr1' # Dummy chromosome
        })
        dummy_regions_df.to_parquet(os.path.join(data_dir, 'regions.parquet'))

    if not os.path.exists(os.path.join(data_dir, 'ensemble_annotation.gff3')):
        print("Creating dummy ensemble_annotation.gff3 for demonstration...")
        # Minimal GFF3 content
        gff_content = """##gff-version 3
chr1\tENSEMBL\tgene\t100\t200\t.\t+\t.\tID=gene001;Name=GeneA
chr1\tENSEMBL\texon\t120\t180\t.\t+\t.\tParent=gene001
chr1\tENSEMBL\tgene\t300\t450\t.\t-\t.\tID=gene002;Name=GeneB
chr1\tENSEMBL\tmRNA\t310\t440\t.\t-\t.\tID=mrna002;Parent=gene002
"""
        with open(os.path.join(data_dir, 'ensemble_annotation.gff3'), 'w') as f:
            f.write(gff_content)


    # 2. Initialize the Dataset
    print("\nInitializing GenomeExpressionDataset...")
    try:
        dataset = GenomeExpressionDataset(data_dir)
        print(f"Dataset initialized successfully! Number of samples (regions): {len(dataset)}")
        print(f"\nFirst 5 rows of regions_df:\n{dataset.regions_df.head()}")
        print(f"\nFirst 5 rows of gff_df:\n{dataset.gff_df.head()}")
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        exit()

    # 3. Create a DataLoader
    batch_size = 8
    num_workers = 0 # Set to 0 for Windows or if experiencing multiprocessing issues.
    # For Linux/macOS, you can increase this for faster loading.

    print(f"\nCreating DataLoader with batch_size={batch_size}, num_workers={num_workers}...")
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print("DataLoader created.")

    # 4. Iterate and explore a few batches
    print("\nExploring data from the DataLoader...")
    for i, (sequences, labels, infos) in enumerate(data_loader):
        print(f"\n--- Batch {i+1} ---")
        print(f"Sequences batch shape: {sequences.shape}") # (Batch_size, Sequence_length, One-hot_dim)
        print(f"Labels batch shape: {labels.shape}") # (Batch_size)

        # Print info for the first sample in the batch
        print("\nRegion Info for the first sample in batch:")
        for key, value in infos.items():
            print(f"  {key}: {value[0]}") # Print the value for the first sample

        if i >= 1: # Explore 2 batches and then stop
            break

    print("\n--- data_loader.py Example Finished ---")