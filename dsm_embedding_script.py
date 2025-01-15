import os
import sys
import evolvepro.plm.esm.extract as esm_extract
from evolvepro.src.process import process_dataset

os.makedirs('output', exist_ok=True)
os.makedirs('output/dms', exist_ok=True)
os.makedirs('output/dms/embeddings', exist_ok=True)

# processing raw data into fasta files
project_root = ''
file_path = os.path.join(project_root, 'data' , 'dms', 'activity', 'Source.xlsx')
dataset_name = 'jones'
wt_fasta_path = os.path.join(project_root, 'data' , 'dms', 'wt_fasta', 'jones_WT.fasta')
activity_column = 'DMS_0.625'
cutoff_value = 2.8
output_dir = os.path.join(project_root, 'output' , 'dms')
sheet_name = 'ADRB2'
cutoff_rule = 'greater_than'
cutoff_percentiles = [90, 95]
AA_shift = None
drop_columns = True

brenan_df, brenan_frac = process_dataset(
    file_path=file_path,
    dataset_name=dataset_name,
    wt_fasta_path=wt_fasta_path,
    activity_column=activity_column,
    cutoff_value=cutoff_value,
    output_dir=output_dir,
    sheet_name=sheet_name,
    cutoff_rule=cutoff_rule,
    cutoff_percentiles=cutoff_percentiles,
    AA_shift=AA_shift,
    drop_columns=drop_columns
)

# simulating system command-line input for arguments
sys.argv = ['evolvepro/plm/esm/extract.py', 
            '/./vast/projects/G000448_Protein_Design/model_weights/esm2_t48_15B_UR50D.pt', # model path
            'output/dms/jones.fasta', # input sequence fasta file
            'output/dms/embeddings', # output location
            '--toks_per_batch', '512',
            '--include', 'mean',
            '--concatenate_dir', 'output'
            ]
esm_extract.main()
