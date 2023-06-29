import json, tqdm, collections, re, sys
import itertools
import spacy
import pytextrank
import networkx as nx
import numpy as np

from nltk.corpus import stopwords

from collections import namedtuple, Counter

FactInst = namedtuple('FactInst', 'idx title text entities')
EntInst = collections.namedtuple('EntInst', 'sidx text cleanText label')
EdgeInst = collections.namedtuple('EdgeInst', 'id f1_idx f2_idx f1_entity f2_entity similarity')

STOP_WORDS = stopwords.words('english')+["'s"]
UNIQUE_NE=['PERSON', 'FAC', 'ORG', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW']

def remove_noise(text):
    words = [w.lower() for w in text.split() if w.lower() not in STOP_WORDS]
    clean_text = [re.sub(r"'s|\W|\s", "", word) for word in words]
    return clean_text

def modify_entity(orig_text, sidx=None):
    modi_text = re.sub(r"^\W+", "", orig_text)
    if sidx:
        sidx += len(orig_text) - len(modi_text)
    modi_text = re.sub(r"\W+$", "", modi_text)

    return modi_text, sidx

#### extract key phrases by using PyTextRank
def get_key_phrases(sent, nlp, entities):
    phrases = []
    for phrase in nlp(sent)._.phrases:
        phrases += phrase.chunks
    phrases = sorted(list(set([str(p) for p in phrases])), key=lambda x:len(x), reverse=True)

    for phrase in phrases:
        phrase, _ = modify_entity(phrase)
        for cid in range(len(sent)):
            if sent[cid:].startswith(phrase):
                cleanText = remove_noise(phrase)
                if len(cleanText) > 0:
                    if phrase not in [ent.text for ent in entities]:
                        entities.append(EntInst(sidx=cid,
                                                text=phrase,
                                                cleanText=cleanText,
                                                label='KEY'))

    return entities

def get_entities(sent, nlp):
    entities = []

    named_entities = []
    for ent in nlp(sent).ents:
        ent_text, sidx = modify_entity(ent.text, ent.start_char)
        named_entities.append((sidx, ent_text, ent.label_))

    named_entities = sorted(list(set(named_entities)), key=lambda x: len(x[1]), reverse=True)
    for sidx, entity, label in named_entities:
        cleanText = remove_noise(entity)
        if len(cleanText) > 0:
            if entity not in [ent.text for ent in entities]:
                entities.append(EntInst(sidx=sidx,
                                        text=entity,
                                        cleanText=cleanText,
                                        label=label))
    
    entities = get_key_phrases(sent, nlp, entities)

    return entities

def compute_iou(tokens1, tokens2):
    count1 = Counter(tokens1)
    count2 = Counter(tokens2)

    intersection = sum((count1&count2).values())
    union = sum((count1|count2).values())

    if union == 0:
        return 0
    else:
        return intersection/union

def generate_all_spanning_trees(G):
    all_edges = G.edges.data('weight')
    all_edge_combinations = itertools.combinations(all_edges, len(G) - 1)

    all_maximum_spanning_trees = []
    tree_maximum_score = 0
    for edges in all_edge_combinations:
        T = nx.Graph()
        for n1, n2, weight in edges:
            T.add_edge(n1, n2, weight=weight)
        if nx.is_tree(T):
            score = sum([e[2] for e in T.edges.data('weight')])
            if score > tree_maximum_score:
                all_maximum_spanning_trees = [T]
                tree_maximum_score = score
            elif score == tree_maximum_score:
                all_maximum_spanning_trees.append(T)

    return all_maximum_spanning_trees

def half_sigmoid(x):
    return 2 / (1 +np.exp(x*0.01))

if __name__ == "__main__":
    split = sys.argv[1]
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank")

    fail = 0
    with open(f"musique_ans_supp_{split}.json", "r") as fin:
        data_list = json.load(fin)

    new_data_list = []
    for data in tqdm.tqdm(data_list):
        _id = data["id"]
        add_this = False

        idx_to_fact = dict()
        facts = []
        for paragraph in data["paragraphs"]:
            text = f'{paragraph["title"]}. {paragraph["paragraph_text"]}'
            entities = get_entities(text, nlp)
            factInst = FactInst(idx=paragraph["idx"],
                                title=paragraph["title"],
                                text=text,
                                entities=entities)
            facts.append(factInst)
            idx_to_fact[paragraph["idx"]] = text

        G = nx.Graph()
        all_edges = dict()
        maximum_edges = dict()
        edge_id = 0
        for i in range(len(facts)-1):
            f1_idx = facts[i].idx
            f1_entities = facts[i].entities
            for j in range(i+1, len(facts)):
                f2_idx = facts[j].idx
                maximum_edges[(f1_idx, f2_idx)] = {"similarity": 0, "edges": []}
                
                f2_entities = facts[j].entities
                for f1_ent in f1_entities:
                    for f2_ent in f2_entities:
                        iou = compute_iou(f1_ent.cleanText, f2_ent.cleanText)
                        if iou > 0:
                            edgeInst = EdgeInst(id=edge_id,
                                                f1_idx=f1_idx,
                                                f2_idx=f2_idx,
                                                f1_entity=f1_ent,
                                                f2_entity=f2_ent,
                                                similarity=iou)
                            all_edges[edge_id] = edgeInst
                            edge_id += 1
                            if maximum_edges[(f1_idx, f2_idx)]["similarity"] < iou:
                                maximum_edges[(f1_idx, f2_idx)] = {"similarity": iou, "edges": [edgeInst]}
                            elif maximum_edges[(f1_idx, f2_idx)]["similarity"] == iou:
                                maximum_edges[(f1_idx, f2_idx)]["edges"].append(edgeInst)

                G.add_edge(f1_idx, f2_idx, weight=maximum_edges[(f1_idx, f2_idx)]["similarity"])

        if _id.startswith("2hop"):
            if len(all_edges) == 0:
                data["tree"] = None
                data["bridges"] = None
                new_data_list.append(data)
                fail += 1
                continue

            bridge = sorted(list(all_edges.values()), key=lambda x: x.similarity, reverse=True)[0]
            entity_per_connection = dict()
            entity_per_connection[bridge.f1_idx] = [{"neighbor_fact": bridge.f2_idx,
                                                      "start_index": bridge.f1_entity.sidx,
                                                      "text": bridge.f1_entity.text,
                                                      "cleanText": bridge.f1_entity.cleanText}]
            entity_per_connection[bridge.f2_idx] = [{"neighbor_fact": bridge.f1_idx,
                                                      "start_index": bridge.f2_entity.sidx,
                                                      "text": bridge.f2_entity.text,
                                                      "cleanText": bridge.f2_entity.cleanText}]

            data["tree"] = [tuple([fact.idx for fact in facts])]
            data["bridges"] = entity_per_connection
            new_data_list.append(data)
            continue

        Ts = generate_all_spanning_trees(G)
        Tree_meta = []
        for T in Ts:
            neighbors = dict()
            edge_weights = []
            for u, v, weight in T.edges.data('weight'):
                edge_weights.append(weight)
                #print(f'{u}-{v}: {weight}')
                if u in neighbors:
                    neighbors[u].append(v)
                else:
                    neighbors[u] = [v]
                if v in neighbors:
                    neighbors[v].append(u)
                else:
                    neighbors[v] = [u]

            components = []
            for facts in list(T.edges):
                components += list(facts)
            bridge_facts = [f[0] for f in Counter(components).items() if f[1] > 1]

            selected_bridges = dict()
            inner_scores = []
            for f1_idx in bridge_facts:
                entities_per_connected = dict()
                for f2_idx in neighbors[f1_idx]:
                    if (f1_idx, f2_idx) in maximum_edges:
                        pair = (f1_idx, f2_idx)
                        target_f_idx = "first"
                    else:
                        pair = (f2_idx, f1_idx)
                        target_f_idx = "second"
                        
                    if target_f_idx == "first":
                        entities_per_connected[f2_idx] = [{"edge_id": edgeInst.id, "entity_sidx": edgeInst.f1_entity.sidx, "cleanText": edgeInst.f1_entity.cleanText} for edgeInst in maximum_edges[pair]["edges"]]
                    else:
                        entities_per_connected[f2_idx] = [{"edge_id": edgeInst.id, "entity_sidx": edgeInst.f2_entity.sidx, "cleanText": edgeInst.f2_entity.cleanText} for edgeInst in maximum_edges[pair]["edges"]]

                all_combinations = []
                for entities in entities_per_connected.values():
                    if len(all_combinations) == 0:
                        for ent in entities:
                            all_combinations.append({"distance": 0, "entities": [ent]})
                    else:
                        new_all_combinations = []
                        for subtree in all_combinations:
                            for ent in entities:
                                min_dist = min([abs(sub_ent["entity_sidx"]-ent["entity_sidx"]) for sub_ent in subtree["entities"]])
                                max_iou = max([compute_iou(sub_ent["cleanText"], ent["cleanText"]) for sub_ent in subtree["entities"]])
                                if min_dist != 0 and max_iou == 0:
                                    new_all_combinations.append({"distance": subtree["distance"]+min_dist, 
                                                                "entities": subtree["entities"] + [ent]})
                        all_combinations = new_all_combinations

                selected_bridges[f1_idx] = []
                for idx, comb in enumerate(all_combinations):
                    avg_dist = comb["distance"]/len(comb["entities"])
                    inner_score = half_sigmoid(avg_dist)
                    all_combinations[idx] = comb

                    selected_bridges[f1_idx].append({"comb": comb, "inner_score": inner_score})

            all_tree_config = []
            for f_idx in selected_bridges.keys():
                if len(all_tree_config) == 0:
                    for comb_item in selected_bridges[f_idx]:
                        inner_selected_bridges = {f_idx: comb_item}
                        all_tree_config.append(inner_selected_bridges)
                else:
                    new_all_tree_config = []
                    for inner_selected_bridges in all_tree_config:
                        for comb_item in selected_bridges[f_idx]:
                            inner_selected_bridges[f_idx] = comb_item
                            new_all_tree_config.append(inner_selected_bridges)
                    all_tree_config = new_all_tree_config

            for inner_selected_bridges in all_tree_config:
                inner_scores = [item["inner_score"] for item in inner_selected_bridges.values()]
                tree_score = (sum(edge_weights)/len(edge_weights) + sum(inner_scores)/len(inner_scores))/2
                Tree_meta.append({"fact_connections": list(T.edges),
                                  "score": tree_score,
                                  "via_entity_comb": [item["comb"] for item in inner_selected_bridges.values()]})

        for route in sorted(Tree_meta, key=lambda x: x["score"], reverse=True):
            entity_per_connection = dict()
            fact_per_neighbor_facts = dict()

            for comb in route["via_entity_comb"]:
                for edge_info in comb["entities"]:
                    edge_id = edge_info["edge_id"]
                    edge = all_edges[edge_id]

                    if edge.f1_idx not in entity_per_connection:
                        entity_per_connection[edge.f1_idx] = []
                        fact_per_neighbor_facts[edge.f1_idx] = []
                    if edge.f2_idx not in entity_per_connection:
                        entity_per_connection[edge.f2_idx] = []
                        fact_per_neighbor_facts[edge.f2_idx] = []

                    if edge.f2_idx not in fact_per_neighbor_facts[edge.f1_idx]:
                        fact_per_neighbor_facts[edge.f1_idx].append(edge.f2_idx)
                        entity_per_connection[edge.f1_idx].append({"neighbor_fact": edge.f2_idx,
                                                                "start_index": edge.f1_entity.sidx,
                                                                "text": edge.f1_entity.text,
                                                                "cleanText": edge.f1_entity.cleanText})
                        
                    if edge.f1_idx not in fact_per_neighbor_facts[edge.f2_idx]:
                        fact_per_neighbor_facts[edge.f2_idx].append(edge.f1_idx)
                        entity_per_connection[edge.f2_idx].append({"neighbor_fact": edge.f1_idx,
                                                                "start_index": edge.f2_entity.sidx,
                                                                "text": edge.f2_entity.text,
                                                                "cleanText": edge.f2_entity.cleanText})
                    
            entity_overlap = False
            for entities in entity_per_connection.values():
                for i in range(len(entities)-1):
                    for j in range(i+1, len(entities)):
                        if compute_iou(entities[i]["cleanText"], entities[j]["cleanText"]) > 0.5:
                            entity_overlap = True
            
            if entity_overlap:
                continue

            #print(route["fact_connections"])
            #print(entity_per_connection)

            #for fact_idx, connections in entity_per_connection.items():
            #    for connection in connections:
            #        text = connection["text"]
            #        start_index = connection["start_index"]
            #        print(text)
            #        fact = idx_to_fact[fact_idx]
            #        print(fact[start_index:start_index+len(text)])
            #        print("-"*30)
            #print("#"*30)


            if len(data["paragraphs"]) != len(entity_per_connection.keys()):
                continue

            data["tree"] = route["fact_connections"]
            data["bridges"] = entity_per_connection
            new_data_list.append(data)
            add_this = True
            break

        if not add_this:
            data["tree"] = None
            data["bridges"] = None 
            new_data_list.append(data)
            fail += 1       
                
    print("total:", len(new_data_list))
    print("fail:", fail)
    with open(f"musique_ans_supp_{split}_bridge.json", "w") as fout:
        json.dump(new_data_list, fout, indent=1)




        

