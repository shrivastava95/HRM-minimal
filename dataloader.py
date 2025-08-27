from huggingface_hub import snapshot_download
import os
import csv
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
from torch.utils.data import Dataset, DataLoader

def download_sudoku_dataset():
    """
    Downloads the Sudoku dataset from the Hugging Face Hub and saves it to a local directory.
    """
    # Define the repository ID and local download path
    repo_id = "sapientinc/sudoku-extreme"
    local_download_path = "data/sudoku-extreme-full"
    
    # Create the local directory if it doesn't exist
    os.makedirs(local_download_path, exist_ok=True)
    
    print(f"Downloading dataset to {local_download_path}...")
    # Download the dataset
    snapshot_download(
        repo_id=repo_id,repo_type="dataset",
        local_dir=local_download_path
    )
    print("Download complete.")
    return local_download_path


class SudokuDataset(Dataset):
    """
    A PyTorch Dataset for loading Sudoku puzzles from a CSV file.
    """
    def __init__(self, config, csv_path, enforce_global_batch_size=True):
        """
        Args:
            csv_path (str): The path to the csv file containing the sudoku puzzles.
        """
        self.config = config
        self.enforce_global_batch_size = enforce_global_batch_size
        self.quizzes = []
        self.solutions = []
        
        print(f"Loading data from {csv_path}...")
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader) # Skip header
            for row in tqdm(reader):
                quiz_str = row[1]
                solution_str = row[2]
                

                quiz = np.array(list(quiz_str.replace('.','0')), dtype=int)
                quiz[quiz==1] = 10
                quiz[quiz==0] = 1
                # quiz = np.array(list)
                solution = np.array(list(solution_str), dtype=int)
                solution[solution==1] = 10
                # breakpoint()
                self.quizzes.append(quiz)
                self.solutions.append(solution)
        print(f"Loaded {len(self.quizzes)} puzzles.")

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.quizzes)

    def __getitem__(self, idx):
        """
        Fetches the sample at the given index.
        
        Args:
            idx (int): The index of the sample to fetch.
            
        Returns:
            tuple: (quiz, solution) where both are torch tensors.
        """
        quiz = torch.from_numpy(self.quizzes[idx]).to(torch.int32)
        solution = torch.from_numpy(self.solutions[idx]).to(torch.int32)
        puzzle_identifier = torch.tensor(0, dtype= torch.int32)
        return {"inputs":quiz,"labels": solution,"puzzle_identifiers": puzzle_identifier}
    
    def __collate_fn__(self, batch):
        # the deafult collate function, if i had not defined this here, would
        if self.enforce_global_batch_size:
            local_batch_size = len(batch)
            if local_batch_size < self.config.batch_size:
                # give the user a warning
                print(f"Warning: batch size {local_batch_size} is less than global batch size {self.config.batch_size}. This is likely due to the last batch of the dataset. Please set drop_last=False in `create_dataloader`, or modify the padding behaviour of the last batch.")
                print(f"Right now, the batch will be padded with config.ignore_index={self.config.ignore_index}")
                # repeat the batch as many times as needed to reach the global batch size.
                # this makes a secondary fallback mechanism for the desired behaviour of the last batch.
                batch = [{k: v.clone() for k, v in batch[_i % local_batch_size].items()} for _i in range(self.config.batch_size)]
                for _i in range(local_batch_size, self.config.batch_size):
                    batch[_i]["labels"][:] = self.config.ignore_index
        # stack the batch
        
        batch = {
            'inputs': torch.stack([x['inputs'] for x in batch]),
            'labels': torch.stack([x['labels'] for x in batch]),
            'puzzle_identifiers': torch.stack([x['puzzle_identifiers'] for x in batch]),
        }
        return batch


def create_dataloader(dataset, batch_size, shuffle=True, drop_last=False):
    """
    Creates a DataLoader for the given dataset.
    
    Args:
        dataset (Dataset): The dataset to create a DataLoader for.
        batch_size (int): The number of samples per batch.
        shuffle (bool): Whether to shuffle the data at every epoch.
        
    Returns:
        DataLoader: A PyTorch DataLoader for the dataset.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=dataset.__collate_fn__)



