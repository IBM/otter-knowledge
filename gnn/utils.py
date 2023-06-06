# utils function for gnn
import random

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# random.seed(42)


def negative_relation_sampling(triples,
                               entity_entity_relation_ids,
                               existing_rel_ratio=0.7,
                               non_existing_rel_ratio=0.3):
    all_triples = set()
    entity_entity_relation_ids_set = set(entity_entity_relation_ids)
    num_triples = triples.size(-1)
    existing_subject_object_pairs = set()
    for idx in range(num_triples):
        s, p, o = triples[0, idx].item(), triples[1, idx].item(), triples[2, idx].item()
        existing_subject_object_pairs.add((s, o))
        all_triples.add((s, p, o))

    # create negative samples from existing relations
    negative_triples = []
    for idx in range(num_triples):
        s, p, o = triples[0, idx].item(), triples[1, idx].item(), triples[2, idx].item()
        if p in entity_entity_relation_ids_set:
            # generate a new relation for s and o
            r = random.choice(entity_entity_relation_ids)
            if (s, r, o) not in all_triples and (o, r, s) not in all_triples:
                if random.random() <= existing_rel_ratio:
                    negative_triples.append([s, r, o])

    # create negative samples from non-relations triples
    sampled_triple_indices = list(range(num_triples))
    random.shuffle(sampled_triple_indices)
    for i in range(num_triples):
        j = sampled_triple_indices[i]
        s, o = triples[0, i].item(), triples[2, j].item()
        r = random.choice(entity_entity_relation_ids)
        if (s, o) not in existing_subject_object_pairs and (o, s) not in existing_subject_object_pairs:
            if random.random() <= non_existing_rel_ratio:
                negative_triples.append([s, r, o])
    return negative_triples


def negative_property_sampling(triples, entity_property_relation_ids, sampling_ratio=1.0):
    num_triples = triples.size(-1)
    # group triples by relations
    groups = {}
    negative_triples = []
    for idx in range(num_triples):
        s, p, o = triples[0, idx].item(), triples[1, idx].item(), triples[2, idx].item()
        if p in entity_property_relation_ids:
            if p not in groups:
                groups[p] = [(s, p, o)]
            else:
                groups[p].append((s, p, o))
    for v in groups.values():
        for s, p, o in v:
            property_idx = random.randint(0, len(v) - 1)
            if v[property_idx][2] != o:
                if random.random() <= sampling_ratio:
                    negative_triples.append((s, p, v[property_idx][2]))
    return negative_triples


def get_push_pull_ids(index_map, one_hop_nodes):
    push_id = []
    pull_id = []
    history_push_id = []
    history_pull_id = []
    for idx, map_idx in index_map.items():
        if idx not in one_hop_nodes:
            history_push_id.append(idx)
            push_id.append(index_map[idx])
        else:
            pull_id.append(index_map[idx])
            history_pull_id.append(idx)
    return torch.tensor(push_id).to(device), torch.tensor(pull_id).to(device), \
           torch.tensor(history_push_id), torch.tensor(history_pull_id)


def split_triples(triples, split_ratios):
    num_triples = triples.size(1)
    indices = list(range(num_triples))
    random.shuffle(indices)
    num_train = int(split_ratios[0]*num_triples)
    num_val = int(split_ratios[1] * num_triples)
    train_indices = torch.tensor(indices[: num_train])
    val_indices = torch.tensor(indices[num_train: (num_train+num_val)])
    test_indices = torch.tensor(indices[num_train+num_val:])
    train_triples = torch.index_select(triples, 1, train_indices)
    val_triples = torch.index_select(triples, 1, val_indices)
    test_triples = torch.index_select(triples, 1, test_indices)
    return train_triples, val_triples, test_triples


def remove_direction(triples):
    visited = {}
    undirected_triples = []
    for i in range(triples.size(1)):
        s, r, t = triples[0][i].item(), triples[1][i].item(), triples[2][i].item()
        visited[(s, r, t)] = 1
        undirected_triples.append([s, r, t])

    for i in range(triples.size(1)):
        s, r, t = triples[0][i].item(), triples[1][i].item(), triples[2][i].item()
        if (t, r, s) not in visited:
            undirected_triples.append([t, r, s])

    undirected_triples = torch.tensor(undirected_triples).permute(1, 0)
    return undirected_triples


def clean_modality_name(modality):
    clean_modality = modality.replace('.', '-')
    return clean_modality


def mask_triples(triples, masking_ratio):
    num_triples = triples.size(1)
    indices = list(range(num_triples))
    random.shuffle(indices)
    num_train = int((1-masking_ratio)*num_triples)
    train_indices = indices[: num_train]
    if len(train_indices) == 0:
        train_indices = [0]
    train_indices = torch.tensor(train_indices).to(device)
    val_indices = indices[num_train:]
    if len(val_indices) == 0:
        val_indices = [0]
    val_indices = torch.tensor(val_indices).to(device)
    train_triples = torch.index_select(triples, 1, train_indices)
    val_triples = torch.index_select(triples, 1, val_indices)
    return train_triples.to(device), val_triples.to(device)


def get_target_triples(triples, masking_ratio):
    target_triples = set()
    group = group_triples(triples)
    num_triples = len(group)
    indices = list(range(num_triples))
    random.shuffle(indices)
    num_train = int((1-masking_ratio)*num_triples)
    val_indices = indices[num_train:]
    for i in val_indices:
        k, v = group[i]
        s, r, o = k
        target_triples.add((s, r, o))
        if v == 2:
            target_triples.add((o, r, s))
    return target_triples


def group_triples(triples):
    group = {}
    for i in range(triples.size(1)):
        s, r, o = triples[0][i].item(), triples[1][i].item(), triples[2][i].item()
        if (s, r, o) not in group and (o, r, s) not in group:
            group[(s, r, o)] = 1
        elif (s, r, o) in group:
            group[(s, r, o)] = 2
        else:
            group[(o, r, s)] = 2
    group = [(k, v) for k, v in group.items()]
    return group


def get_partition_mask_triples(triples, all_target_triples, index_map):
    inverse_index_map = {}
    for k, v in index_map.items():
        inverse_index_map[v] = k
    train_indices = []
    target_indices = []
    for i in range(triples.size(1)):
        s, r, o = triples[0][i].item(), triples[1][i].item(), triples[2][i].item()
        si, ri, oi = inverse_index_map[s], r, inverse_index_map[o]
        if (si, ri, oi) in all_target_triples:
            target_indices.append(i)
        else:
            train_indices.append(i)
    if len(target_indices) == 0:
        target_indices.append(0)
    train_indices = torch.tensor(train_indices).to(device)
    target_indices = torch.tensor(target_indices).to(device)
    train_triples = torch.index_select(triples, 1, train_indices)
    target_triples = torch.index_select(triples, 1, target_indices)
    return train_triples.to(device), target_triples.to(device)


def shuffle(perm, ptr):
    partitions = []
    for i in range(len(ptr)-1):
        partitions.append(perm[ptr[i]:ptr[i+1]])
    random.shuffle(partitions)
    new_ptr = [0]
    p = 0
    for i in range(len(partitions)):
        p += len(partitions[i])
        new_ptr.append(p)
    new_ptr = torch.tensor(new_ptr)
    new_perm = torch.cat(partitions)
    return new_perm, new_ptr


def get_neg_triples_old(all_triples, triples, index_map, regression_rel, sampling_ratio=1.0):
    # random.seed(42)
    inverse_index_map = {}
    for k, v in index_map.items():
        inverse_index_map[v] = k
    num_triples = triples.size(-1)
    # group triples by relations
    groups = {}
    negative_triples = []
    for idx in range(num_triples):
        s, p, o = triples[0, idx].item(), triples[1, idx].item(), triples[2, idx].item()
        if p not in groups:
            groups[p] = [(s, p, o)]
        else:
            groups[p].append((s, p, o))

    # loop through groups and create negative triples
    for p, v in groups.items():
        so_far = set()
        num_checks = int(len(v)*sampling_ratio)
        for i in range(num_checks):
            # sample a random object
            random_o_idx = random.randint(0, len(v) - 1)
            random_s_idx = random.randint(0, len(v) - 1)
            random_o = v[random_o_idx][2]
            random_s = v[random_s_idx][0]
            si, ri, oi = inverse_index_map[random_s], p, inverse_index_map[random_o]
            if random.random() <= sampling_ratio \
                    and (si, p, oi) not in all_triples \
                    and (oi, p, si) not in all_triples \
                    and (random_s, p, random_o) not in so_far \
                    and (random_o, p, random_s) not in so_far:
                if regression_rel is None or p not in regression_rel:
                    negative_triples.append([random_s, p, random_o])
                    so_far.add((random_s, p, random_o))
    if len(negative_triples) == 0:
        negative_triples.append([random_s, p, random_o])
    negative_triples = torch.tensor(negative_triples)
    negative_triples = negative_triples.permute(1, 0)
    return negative_triples.to(device)


def get_neg_triples(all_triples, triples, index_map, regression_rel, node_modalities, sampling_ratio=1.0):
    # random.seed(42)
    inverse_index_map = {}
    for k, v in index_map.items():
        inverse_index_map[v] = k
    num_triples = triples.size(-1)
    # group triples by relations
    groups = {}
    groups_modality = {}
    negative_triples = []
    for idx in range(num_triples):
        s, p, o = triples[0, idx].item(), triples[1, idx].item(), triples[2, idx].item()
        si, oi = inverse_index_map[s], inverse_index_map[o]
        s_modality = node_modalities[si]
        o_modality = node_modalities[oi]
        if p not in groups_modality:
            groups_modality[p] = (s_modality, o_modality)
        if p not in groups:
            groups[p] = [(s, p, o)]
        else:
            groups[p].append((s, p, o))
    modality_nodes = {}
    for k, v in inverse_index_map.items():
        modality = node_modalities[v]
        if modality not in modality_nodes:
            modality_nodes[modality] = [k]
        else:
            modality_nodes[modality].append(k)

    # loop through groups and create negative triples
    for p, v in groups.items():
        so_far = set()
        num_checks = int(len(v)*sampling_ratio)
        s_modality, o_modality = groups_modality[p][0],  groups_modality[p][1]
        for i in range(num_checks):
            # sample a random object
            random_o_idx = random.randint(0, len(modality_nodes[o_modality]) - 1)
            random_s_idx = random.randint(0, len(modality_nodes[s_modality]) - 1)
            random_o = modality_nodes[o_modality][random_o_idx]
            random_s = modality_nodes[s_modality][random_s_idx]
            si, ri, oi = inverse_index_map[random_s], p, inverse_index_map[random_o]
            if random.random() <= sampling_ratio \
                    and (si, p, oi) not in all_triples \
                    and (oi, p, si) not in all_triples \
                    and (random_s, p, random_o) not in so_far \
                    and (random_o, p, random_s) not in so_far:
                if regression_rel is None or p not in regression_rel:
                    negative_triples.append([random_s, p, random_o])
                    so_far.add((random_s, p, random_o))
    if len(negative_triples) == 0:
        negative_triples.append([random_s, p, random_o])
    negative_triples = torch.tensor(negative_triples)
    negative_triples = negative_triples.permute(1, 0)
    return negative_triples.to(device)


def get_modality(dataset):
    node_modalities = {}
    for n, v in dataset.graph_cfg.numeric_id_node_id_map.items():
        node_modalities[n] = dataset.nodes[v].modality
    return node_modalities


def get_node_number_values(node_data, triples, dataset, index_map):
    values = {}
    inverse_index_map = {}
    for k, v in index_map.items():
        inverse_index_map[v] = k
    if 'number' not in node_data:
        return None, None
    for i, v in zip(node_data['number']['node_indices'], node_data['number']['embeddings']):
        values[i.item()] = v
    y = []
    new_triples = []
    for j in range(len(triples[0])):
        s, r, o = triples[0][j].item(), triples[1][j].item(), triples[2][j].item()
        if s not in values:
            si = inverse_index_map[s]
            sid = dataset.graph_cfg.numeric_id_node_id_map[si]
            print("Warning: inconsistent modality", sid, dataset.nodes[sid].content,
                  dataset.nodes[sid].modality)
        else:
            y.append(values[s])
            new_triples.append([s, r, o])
    if len(y) > 0:
        y = torch.stack(y)
        new_triples = torch.tensor(new_triples).permute(1, 0)
        return y.to(device), new_triples.to(device)
    else:
        return None, None


def get_only_enable_triples(triples, graph_cfg, enable_relation):
    if len(enable_relation) > 0:
        interesting_relation_ids = [graph_cfg.relation_numeric_id_map[r] for r in enable_relation]
        idx = sum(triples[1] == i for i in interesting_relation_ids).bool().to(device)
        return triples[:, idx], triples[:, ~idx]
    else:
        return triples


def print_triples_stats(triples, index_map, graph_cfg):
    print("Triple stats")
    inverse_index_map = {}
    for k, v in index_map.items():
        inverse_index_map[v] = k
    stats = {}
    source_nodes = {}
    target_nodes = {}
    for j in range(triples.size(1)):
        s, r, t = triples[0][j].item(), triples[1][j].item(), triples[2][j].item()
        rid = graph_cfg.numeric_id_relation_map[r]
        if rid not in stats:
            stats[rid] = 1
        else:
            stats[rid] += 1
        if s not in source_nodes:
            source_nodes[s] = 1
        else:
            source_nodes[s] += 1
        if t not in target_nodes:
            target_nodes[t] = 1
        else:
            target_nodes[t] += 1
    print(stats)
    simple_node_stats(source_nodes)
    simple_node_stats(target_nodes)


def simple_node_stats(nodes):
    mi = None
    mc = None
    for i, c in nodes.items():
        if mc is None or mc < c:
            mi = i
            mc = c
    print("Node stats", len(nodes), mi, mc)


def print_triples(triples, index_map, graph_cfg):
    print("Triples --------------------------")
    inverse_index_map = {}
    for k, v in index_map.items():
        inverse_index_map[v] = k
    for j in range(triples.size(1)):
        s, r, t = triples[0][j].item(), triples[1][j].item(), triples[2][j].item()
        rid = graph_cfg.numeric_id_relation_map[r]
        sid = graph_cfg.numeric_id_node_id_map[s]
        tid = graph_cfg.numeric_id_node_id_map[t]
        print(sid, rid, tid)


def get_triples_stats(triples, index_map, graph_cfg, stats):
    inverse_index_map = {}
    m = 0
    n = 0
    for k, v in index_map.items():
        inverse_index_map[v] = k
    for j in range(triples.size(1)):
        s, r, t = triples[0][j].item(), triples[1][j].item(), triples[2][j].item()
        rid = graph_cfg.numeric_id_relation_map[r]
        sid = graph_cfg.numeric_id_node_id_map[s]
        tid = graph_cfg.numeric_id_node_id_map[t]
        if (sid, rid, tid) not in stats:
            stats[(sid, rid, tid)] = 1
        else:
            stats[(sid, rid, tid)] += 1
    for k, v in stats.items():
        if v > 1:
            n += 1
        if m < v:
            m = v
    print("Max, count", m, n)



