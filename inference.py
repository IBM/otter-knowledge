import argparse
import json
import os.path
import time
from typing import Union, Optional

import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm

from data_utils.dataset import InputDataset
from embeddings.esm import ESMProtein
from embeddings.morgan_fingerprint import MorganFingerprint
from models.load_from_hub import load_from_hub


def get_embeddings(sequences, net, initial_model, relation_map,
                   input_type: Optional[str] = "Protein",
                   device: Optional[Union[str, torch.device]] = 'cpu'):
    initial_embeddings = initial_model.get_embeddings(sequences)
    if input_type == 'Protein':
        modality = 'protein-sequence-mean'
        entity_name = 'Protein'
        rel_id = relation_map['sequence']
    else:
        modality = 'morgan-fingerprint'
        entity_name = 'Drug'
        rel_id = relation_map['smiles']

    output_embeddings = []
    for embeddings in initial_embeddings:
        # create nodes for scoring
        nodes = {
            modality: {
                'embeddings': embeddings.unsqueeze(0).to(device),
                'node_indices': torch.tensor([0]).to(device)
            },
            entity_name: {
                'embeddings': [None],
                'node_indices': torch.tensor([1]).to(device)
            }
        }

        triples = torch.tensor([[0], [rel_id], [1]]).to(device)
        with torch.no_grad():
            node_output_embeddings = net.encoder(nodes,
                                                 triples)
            output_embeddings.append(node_output_embeddings[1])
    return output_embeddings, initial_embeddings


if __name__ == '__main__':
    # %% getting parameters
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--input_path', required=True,
                        type=str,
                        help='Path to the csv file with the sequence/smiles')
    parser.add_argument('--sequence_column', default='Target', type=str,
                        help='Name of the column with sequence/smiles information for proteins or molecules')
    parser.add_argument('--input_type', default='Protein', type=str,
                        help='Type of the sequences. Options: Drug; Protein')
    parser.add_argument('--model_path', default='ibm/otter_ubc_distmult', type=str,
                        help='Path to the model or name of the model in the HuggingfaceHub')
    parser.add_argument('--output_path', required=True, type=str,
                        help='Path to the output embedding file.')
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size to use.')
    parser.add_argument('--no_cuda', action="store_true", help="If set to True, CUDA won't be used even if available.")

    args, unknown = parser.parse_known_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print("DEVICE", device)

    if args.input_type == 'Protein':
        initial_model = ESMProtein(repo_or_dir='facebookresearch/esm:main',
                                   model='esm1b_t33_650M_UR50S',
                                   repr_layer=33, device=device)
    else:
        initial_model = MorganFingerprint()

    start = time.time()
    if os.path.exists(args.model_path) and os.path.isdir(args.model_path):
        model_path = os.path.join(args.model_path, "model.pt")
        relation_map_path = os.path.join(args.model_path, "relation_map.json")
        print("Loading model from", model_path)
        net = torch.load(model_path, map_location=torch.device(device))
        # relation map read from the json file in the checkpoint path
        with open(relation_map_path) as f:
            print("Loading relation map from", relation_map_path)
            relation_map = json.load(f)
    else:
        print("Path not found, trying to download it from the Hub:")
        model_file, relation_map = load_from_hub(args.model_path)
        net = torch.load(model_file, map_location=torch.device(device))
        with open(relation_map) as f:
            relation_map = json.load(f)

    net.eval()
    embeddings = {
        'Drug': {},
        'Target': {}
    }
    # load test data from a csv file
    test_data = InputDataset(args.input_path, args.sequence_column)
    test_data = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    for data in tqdm(test_data):
        embedding, initial_embeddings = get_embeddings(sequences=data, net=net,
                                                       initial_model=initial_model,
                                                       relation_map=relation_map,
                                                       input_type=args.input_type,
                                                       device=device)
        for s, e, ie in zip(data, embedding, initial_embeddings):
            if args.input_type == 'Protein':
                embeddings['Target'][s] = torch.concat([e, ie]).cpu().tolist()
            else:
                embeddings['Drug'][s] = torch.concat([e, ie]).cpu().tolist()

    output_name = args.output_path
    print('Saving embeddings to', output_name)
    with open(output_name, 'wt') as f:
        json.dump(embeddings, f, indent=4)
