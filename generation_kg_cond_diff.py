import torch
from models.CPDiffusionModel import KGCondDiffusion

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = KGCondDiffusion().load_from_checkpoint(
    "data/output/pepSeqCondDiff/pepSeqCondDiff-XXX.ckpt",
    map_location=device)
# load the trained model

model = model.to(device)
model.eval()

with torch.no_grad():
    out_seq_list, out_seq_traj, record_path = model.denoise_seq_sample(
        n_seq=2000,
        # output sequence numbers
        pathogen_type="pathogen type",
        # given pathogen type; like "S.aureus","E.faecalis", "E.coli","A.baumannii"
        fasta_out_statue=True,
        # save the output sequences as fasta file
        file_name="file_name"
        # output file name
    )


