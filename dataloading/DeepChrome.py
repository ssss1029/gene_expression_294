"""

Default DeepChrome Dataloader

"""

import csv
import torch
import glob
import logging

import multiprocessing
import numpy as np

class DeepChromeDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, num_procs=24):
        self.dataroot = dataroot # Should be a list of glob strings.
        self.num_procs = num_procs
        self.samples = [] # List of tuples (torch.Tensor[100x5], torch.Tensor[1])

        self._load_from_dataroot()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return {
            'X' : torch.from_numpy(sample['X']).float(),
            'y' : torch.from_numpy(sample['y']).squeeze(-1),
            'gene_id' : sample['gene_id'],
        }

    def _load_from_dataroot(self):
        files = []
        for glob_str in self.dataroot:
            files.extend(glob.glob(glob_str))
        assert len(files) != 0
        
        # Code is inefficient and I'm lazy.
        with multiprocessing.Pool(self.num_procs) as pool:
            proc_results = pool.map(self._load_file_faster, files)

        for result in proc_results:
            # self.samples is going to contain numpy arrays.
            # Fix this in __getitem__()
            self.samples.extend(result)

    def _load_file(self, fname):
        """
        Pls excuse the shit code.
        """

        # file_contents = {
        #     "gene_id" : {
        #         "bin_id" : [HM1_count, ..., HM5_count]
        #         ...
        #         "expression" : 0/1
        #     }
        # }
        file_contents = dict()

        with open(fname, 'r') as f:
            reader = csv.DictReader(
                f,
                fieldnames=[
                    "gene_id", 
                    "bin_id", 
                    "H3K27me3_count", 
                    "H3K36me3_count", 
                    "H3K4me1_count", 
                    "H3K4me3_count", 
                    "H3K9me3_count", 
                    "gene_expression"
                ]
            )
            
            for row in reader:
                if row['gene_id'] not in file_contents:
                    file_contents[row['gene_id']] = dict()
                
                file_contents[row['gene_id']][row['bin_id']] = [
                    row['H3K27me3_count'],
                    row['H3K36me3_count'],
                    row['H3K4me1_count'],
                    row['H3K4me3_count'],
                    row['H3K9me3_count'],
                ]

                # Sanity check.
                assert file_contents[row['gene_id']].get('expression', None) == None \
                    or file_contents[row['gene_id']].get('expression', None) == row['gene_expression']
                
                file_contents[row['gene_id']]['expression'] = row['gene_expression']
        
        # Now that we have file contents loaded, create X and Y.
        samples = []
        for gene_id, bins in file_contents.items():
            # Sanity check we have 100 bins for each gene_id
            assert len(bins) == 100 + 1 # Add 1 for the expression.

            Y = torch.zeros((1))
            X = torch.zeros((100, 5))
            for key in bins:
                if key == 'expression':
                    Y[0] = int(bins[key])
                else:
                    bin_id = int(key) - 1 # Indices go from 0-99, but the CSV has it in 1-100
                    X[bin_id][0] = float(bins[key][0]) # H3K27me3_count
                    X[bin_id][1] = float(bins[key][1]) # H3K36me3_count
                    X[bin_id][2] = float(bins[key][2]) # H3K4me1_count
                    X[bin_id][3] = float(bins[key][3]) # H3K4me3_count
                    X[bin_id][4] = float(bins[key][4]) # H3K9me3_count
            
            # Convert to numpy here, and caller is responsible for converting back to PyTorch
            # This is because of multiprocessing being weird with PyTorch.
            samples.append({
                "X" : X.numpy(),
                "Y" : Y.numpy(),
                "gene_id" : gene_id
            })

        return samples

    def _load_file_faster(self, fname):
        """
        Pls excuse the shit code.
        """

        samples = dict()

        with open(fname, 'r') as f:
            reader = csv.DictReader(
                f,
                fieldnames=[
                    "gene_id", 
                    "bin_id", 
                    "H3K27me3_count", 
                    "H3K36me3_count", 
                    "H3K4me1_count", 
                    "H3K4me3_count", 
                    "H3K9me3_count", 
                    "gene_expression"
                ]
            )
            
            for row in reader:
                gene_id = row['gene_id']
                bin_id = int(row['bin_id']) - 1
                hm1 = int(row['H3K27me3_count'])
                hm2 = int(row['H3K36me3_count'])
                hm3 = int(row['H3K4me1_count'])
                hm4 = int(row['H3K4me3_count'])
                hm5 = int(row['H3K9me3_count'])
                gene_expression = int(row['gene_expression'])

                if gene_id not in samples:
                    samples[gene_id] = {
                        "X" : np.zeros((100, 5)),
                        "y" : None, 
                        "gene_id" : gene_id
                    }
                
                samples[gene_id]["X"][bin_id][0] = hm1
                samples[gene_id]["X"][bin_id][1] = hm2
                samples[gene_id]["X"][bin_id][2] = hm3
                samples[gene_id]["X"][bin_id][3] = hm4
                samples[gene_id]["X"][bin_id][4] = hm5

                # Sanity check.
                assert samples[gene_id]['y'] == None \
                    or samples[gene_id]['y'][0] == gene_expression
                
                samples[gene_id]['y'] = np.array([gene_expression])
        
        return list(samples.values())

if __name__ == '__main__':
    dset = DeepChromeDataset(
        dataroot="/accounts/projects/jsteinhardt/sauravkadavath/gene_expression_294/dataset/E098/classification/train.csv",
    )

    print(dset[0])

    import pdb; pdb.set_trace()