import os, argparse, tqdm, random, json, spacy, copy
import pytextrank
import matplotlib.pyplot as plt
import numpy as np
import collections

import nltk
from rouge import Rouge

import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset, default_collate

from models import BartForConditionalGeneration as ConditionalGeneration
from transformers import AutoTokenizer

args = None
nlp = None
FeatInst = collections.namedtuple('FeatInst', 'unique_id input_ids attention_mask decoder_input_ids decoder_attention_mask labels hop')

def parse_argument():
    global args
    parser = argparse.ArgumentParser()

    ### Train ###
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--valid_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2023)

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--do_analyze", action="store_true")

    ### Data ###
    parser.add_argument("--train_data_file", type=str, default="/home/seonjeongh/data/HotpotQA/experiment/hotpot_train.json")
    parser.add_argument("--valid_data_file", type=str, default="/home/seonjeongh/data/HotpotQA/experiment/hotpot_dev.json")
    parser.add_argument("--test_data_file", type=str, default="/home/seonjeongh/data/HotpotQA/experiment/hotpot_test.json")

    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--exp_tag", type=str, default="test")
    parser.add_argument("--checkpoint", type=str, default=None)

    parser.add_argument("--model_name", type=str, default="facebook/bart-large")
    parser.add_argument("--max_encoder_length", type=int, default=512)
    parser.add_argument("--max_decoder_length", type=int, default=128)

    parser.add_argument("--start_epochs", type=str, default="0,0,0,0")
    parser.add_argument("--end_epochs", type=str, default="100,100,100,100")
    parser.add_argument("--mix_batch", action="store_true")
    parser.add_argument("--last_mix_batch", action="store_true")
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--paralevel", action="store_true")

    args = parser.parse_args()
    args.start_epochs = dict([(hop+1, int(epoch)) for hop, epoch in enumerate(args.start_epochs.strip().split(","))])
    args.end_epochs = dict([(hop+1, int(epoch)) for hop, epoch in enumerate(args.end_epochs.strip().split(","))])

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_musique_features(data_file, tokenizer, mode):
    with open(data_file, "r") as fin:
        data_list = json.load(fin)

    bos_token_id = tokenizer.eos_token_id if "bart" in args.model_name.lower() else tokenizer.pad_token_id

    hops = {1: 0, 2: 0, 3: 0, 4: 0}
    features_per_hop = {1: [], 2: [], 3: [], 4:[]}
    unique_to_id = dict()
    unique_to_gold = dict()
    unique_id = 0
    passed_input_num, passed_output_num = 0, 0
    for data in tqdm.tqdm(data_list):
        hop = int(data["id"].split("hop")[0])

        pid_to_fact = dict()
        for para in data["paragraphs"]:
            pid = para["idx"]
            fact = f"{para['title']}. {para['paragraph_text']}"
            pid_to_fact[pid] = fact

        ## Extract 1-hop QA
        number_to_blank_answer = dict()
        for i, item in enumerate(data["question_decomposition"]):
            number_to_blank_answer[f"#{i+1}"] = item["answer"]

        for intermediate in data["question_decomposition"]:
            question = intermediate["question"]
            complete_sinlge = True
            if ">>" in question:
                complete_sinlge = False
            ###########################################
            #for number in number_to_blank_answer.keys():
            #    if number in question:
            #        complete_sinlge = False
            ###########################################
            
            if complete_sinlge:
                for number, blank_answer in number_to_blank_answer.items():
                    if number in question:
                        question = question.replace(number, blank_answer)

                _id = intermediate["id"]
                pid = intermediate["paragraph_support_idx"]
                answer = intermediate["answer"]

                fact = pid_to_fact[pid]

                input_seq = "Answer: " + answer + " Fact: " + fact
                encoded_input = tokenizer(input_seq)
                input_ids = encoded_input["input_ids"]
                attention_mask = encoded_input["attention_mask"]

                if len(input_ids) > args.max_encoder_length:
                    print(f"Long Input: {len(input_ids)}")
                    input_ids = input_ids[:args.max_encoder_length]
                    attention_mask = attention_mask[:args.max_encoder_length]
                    
                while len(input_ids) < args.max_encoder_length:
                    input_ids.append(tokenizer.pad_token_id)
                    attention_mask.append(0)

                assert len(input_ids) == args.max_encoder_length
                assert len(attention_mask) == args.max_encoder_length

                if mode == "train":
                    output_seq = question
                    encoded_output = tokenizer(output_seq)
                    labels = encoded_output["input_ids"] + [-100]
                    decoder_input_ids = [bos_token_id] + encoded_output["input_ids"]
                    decoder_attention_mask = [1] + encoded_output["attention_mask"]

                    if len(labels) > args.max_decoder_length:
                        passed_output_num +=1
                        print(f"Long Output ({passed_output_num}): {len(labels)}")
                        continue

                    while len(labels) < args.max_decoder_length:
                        labels.append(-100)
                        decoder_input_ids.append(tokenizer.pad_token_id)
                        decoder_attention_mask.append(0)
                    
                    assert len(decoder_input_ids) == args.max_decoder_length
                    assert len(decoder_attention_mask) == args.max_decoder_length
                    assert len(labels) == args.max_decoder_length
                else:
                    labels = []
                    decoder_input_ids = [bos_token_id]
                    decoder_attention_mask = [1]

                feature = FeatInst(unique_id=unique_id,
                                   input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   decoder_input_ids=decoder_input_ids,
                                   decoder_attention_mask=decoder_attention_mask,
                                   labels=labels,
                                   hop=1)
                features_per_hop[1].append(feature)
                hops[1] += 1
                unique_to_id[unique_id] = _id
                unique_to_gold[unique_id] = question
                unique_id += 1

        ## Orignal Hop
        question = data["question"]
        answer = data["answer"]

        id_to_entities = dict()
        id_to_key_phrases = dict()
        global_intersections = None
        for component in data["question_decomposition"]:
            pid = component["paragraph_support_idx"]
            fact = pid_to_fact[pid]

            total_entities = set()
            entities = set()
            for ent in nlp(fact).ents:
                entities.add(ent.text)
            id_to_entities[pid] = entities
            total_entities = total_entities | entities

            entities = set()
            for phrase in nlp(fact)._.phrases:
                entities.add(phrase.text)
            id_to_key_phrases[pid] = entities
            total_entities = total_entities | entities
            

            if global_intersections is None:
                global_intersections = set()
                global_intersections = global_intersections | total_entities
            else:
                global_intersections = global_intersections & total_entities

        for pid, entities in id_to_entities.items():
            id_to_entities[pid] = entities - global_intersections
            id_to_key_phrases[pid] = id_to_key_phrases[pid] - global_intersections

        pid_to_intersections = dict()
        id_list = id_to_entities.keys()
        for i in id_list:
            pid_to_intersections[i] = dict()
            for j in id_list:
                if i == j:
                    continue
                i_entities = id_to_entities[i]
                j_fact = pid_to_fact[j]
                intersection = []
                for i_ent in i_entities:
                    for i_ent_token in i_ent.split():
                        if i_ent_token in j_fact:
                            intersection.append(i_ent)
                            break
                if len(intersection) == 0:
                    i_entities = id_to_key_phrases[i]
                    j_fact = pid_to_fact[j]
                    intersection = []
                    for i_ent in i_entities:
                        for i_ent_token in i_ent.split():
                            if i_ent_token in j_fact:
                                intersection.append(i_ent)                

                pid_to_intersections[i][j] = set(intersection)

        ans_fact_pid = None
        for intermediate in data["question_decomposition"]:
            if intermediate["answer"] == answer:
                ans_fact_pid = intermediate["paragraph_support_idx"]

        other_fact_pids = list(pid_to_fact.keys())
        other_fact_pids.remove(ans_fact_pid)

        input_seq_list = []

        #### Initial Input
        avail_facts = set()
        bridges = set()
        for pid, ents in pid_to_intersections[ans_fact_pid].items():
            if len(ents):
                bridges = bridges | ents
                avail_facts.add(pid)
        input_seq = "Answer: " + answer + " Fact: " + pid_to_fact[ans_fact_pid] + " Bridges: " + " ".join(list(bridges))
        input_seq_list.append(input_seq)
        used_facts = set([ans_fact_pid])
        
        #### following Inputs
        while len(used_facts) < len(pid_to_fact.keys()):
            if len(list(avail_facts - used_facts)):
                next_fact_id = list(avail_facts - used_facts)[0]
            else:
                next_fact_id = list(set(pid_to_fact.keys()) - used_facts)[0]
            bridges = set()
            for pid, ents in pid_to_intersections[next_fact_id].items():
                if pid not in used_facts and len(ents):
                    bridges = bridges | ents
                    avail_facts.add(pid)
            input_seq = "Fact: " + pid_to_fact[next_fact_id]
            if len(bridges):
                input_seq += " Bridges: " + " ".join(list(bridges))
            input_seq_list.append(input_seq)
            used_facts.add(next_fact_id)

        total_input_ids = []
        total_attention_mask = []
            
        """fact_chain = [ans_fact] + other_facts

        entity_list_per_fact = []
        for fact in fact_chain:
            entity_list = []
            passed_entities = []
            for ent in nlp(fact).ents:
                if ent.label_ in ["PERSON", "FAC", "ORG", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW"]:
                    entity_list.append(ent.text)
                else:
                    passed_entities.append(ent.text)
            for phrase in nlp(fact)._.phrases:
                    entity = phrase.text
                    if entity not in passed_entities:
                        entity_list.append(entity)
            entity_list_per_fact.append(list(set(entity_list)))

        for i, fact in enumerate(fact_chain):
            if i < len(fact_chain)-1:
                bridges = set()
                current_entities = [(set(entity.lower().split()), entity) for entity in entity_list_per_fact[i]]
                for j in range(i+1, len(fact_chain)):
                    next_entities = [(set(entity.lower().split()), entity) for entity in entity_list_per_fact[j]]

                    for cur_ent, entity in current_entities:
                        for next_ent, _ in next_entities:
                            if len(cur_ent&next_ent) > 0:
                                bridges.add(entity+".")
                if i == 0:
                    input_seq = "Answer: " + answer + " Fact: " + fact + " Bridges: " + " ".join(list(bridges))
                else:
                    input_seq = "Fact: " + fact + " Bridges: " + " ".join(list(bridges))                 
            else:
                input_seq = "Fact: " + fact"""
        
        for input_seq in input_seq_list:
            encoded = tokenizer(input_seq)
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            if len(input_ids) > args.max_encoder_length:
                print(f"Long Input: {len(input_ids)}")
                input_ids = input_ids[:args.max_encoder_length]
                attention_mask = attention_mask[:args.max_encoder_length]

            while len(input_ids) < args.max_encoder_length:
                input_ids.append(tokenizer.pad_token_id)
                attention_mask.append(0)

            total_input_ids += input_ids
            total_attention_mask += attention_mask

        assert len(total_input_ids) == args.max_encoder_length * hop, f"{len(total_input_ids)}"
        assert len(total_attention_mask) == args.max_encoder_length * hop, f"{len(total_attention_mask)}"

        if mode == "train":
            output_seq = question
            encoded = tokenizer(output_seq)
            labels = encoded["input_ids"] + [-100]
            decoder_input_ids = [bos_token_id] + encoded["input_ids"]
            decoder_attention_mask = [1] + encoded["attention_mask"]

            if len(labels) > args.max_decoder_length:
                passed_output_num +=1
                print("Long Output:", len(labels))
                continue

            while len(labels) < args.max_decoder_length:
                labels.append(-100)
                decoder_input_ids.append(tokenizer.pad_token_id)
                decoder_attention_mask.append(0)

            assert len(decoder_input_ids) == args.max_decoder_length
            assert len(decoder_attention_mask) == args.max_decoder_length
            assert len(labels) == args.max_decoder_length
        else:
            labels = []
            decoder_input_ids = [bos_token_id]
            decoder_attention_mask = [1]

        feature = FeatInst(unique_id=unique_id,
                           input_ids=total_input_ids,
                           attention_mask=total_attention_mask,
                           decoder_input_ids=decoder_input_ids,
                           decoder_attention_mask=decoder_attention_mask,
                           labels=labels,
                           hop=hop)
        features_per_hop[hop].append(feature)
        hops[hop] += 1
        unique_to_id[unique_id] = data["id"]
        unique_to_gold[unique_id] = question
        unique_id += 1

    print("Passed Num:", passed_input_num, passed_output_num)
    print("Hops:", hops)
    return features_per_hop, unique_to_id, unique_to_gold

def get_hotpot_features_paralevel(data_file, tokenizer, mode):
    with open(data_file, "r") as fin:
        data_list = json.load(fin)

    bos_token_id = tokenizer.eos_token_id if "bart" in args.model_name.lower() else tokenizer.pad_token_id

    levels = {"easy": 0, "medium": 0, "hard": 0}
    features_per_hop = {1: [], 2: []}
    unique_to_id = dict()
    unique_to_gold = dict()
    unique_id = 0
    cut = 0
    unintended_singleHop = 0
    for data in tqdm.tqdm(data_list):
        title_to_lines = dict()
        title_to_context = dict()
        for title, lines in data["context"]:
            title_to_lines[title] = lines
            title_to_context[title] = f'{title}. {"".join(lines)}'

        facts = dict()
        try:
            for title, line_no in data["supporting_facts"]:
                if title in facts:
                    facts[title].append(title_to_lines[title][line_no])
                else:
                    facts[title] = [title_to_lines[title][line_no]]
        except:
            continue

        for title, lines in facts.items():
            facts[title] = "".join(lines)

        question = data["question"]
        answer = data["answer"]

        ans_facts = []
        other_facts = []
        for title, line in facts.items():
            sub_fact = title_to_context[title]
            if answer in line:
                ans_facts.append(sub_fact)
            elif " ".join(answer.split()) in " ".join(line.split()):
                ans_facts.append(sub_fact)
            else:
                other_facts.append(sub_fact)

        if not ans_facts:
            if data["type"] == "comparison":
                ans_facts = other_facts
                other_facts = []
            else:
                print("Cannot find answer span")
                ans_facts = other_facts
                other_facts = []
                if mode == "train":
                    continue

        if len(other_facts) == 0:
            if data["level"] != "easy" and data["type"] == "bridge":
                unintended_singleHop += 1
                continue

            ans_fact = " ".join(ans_facts)

            input_seq = "Answer: " + answer + " Fact: " + ans_fact

            encoded_input = tokenizer(input_seq)
            input_ids = encoded_input["input_ids"]
            attention_mask = encoded_input["attention_mask"]

            if len(input_ids) > args.max_encoder_length:
                cut += 1
                input_ids = input_ids[:args.max_encoder_length]
                attention_mask = attention_mask[:args.max_encoder_length]

            while len(input_ids) < args.max_encoder_length:
                input_ids.append(tokenizer.pad_token_id)
                attention_mask.append(0)

            assert len(input_ids) == args.max_encoder_length
            assert len(attention_mask) == args.max_encoder_length

            if mode == "train":
                output_seq = question
                encoded_output = tokenizer(output_seq)
                labels = encoded_output["input_ids"] + [-100]
                decoder_input_ids = [bos_token_id] + encoded_output["input_ids"]
                decoder_attention_mask = [1] + encoded_output["attention_mask"]

                if len(labels) > args.max_decoder_length:
                    continue

                while len(labels) < args.max_decoder_length:
                    labels.append(-100)
                    decoder_input_ids.append(tokenizer.pad_token_id)
                    decoder_attention_mask.append(0)
                
                assert len(decoder_input_ids) == args.max_decoder_length
                assert len(decoder_attention_mask) == args.max_decoder_length
                assert len(labels) == args.max_decoder_length
            else:
                labels = []
                decoder_input_ids = [bos_token_id]
                decoder_attention_mask = [1]

            feature = FeatInst(unique_id=unique_id,
                               input_ids=input_ids,
                               attention_mask=attention_mask,
                               decoder_input_ids=decoder_input_ids,
                               decoder_attention_mask=decoder_attention_mask,
                               labels=labels,
                               hop=1)
            features_per_hop[1].append(feature)
            unique_to_id[unique_id] = data["_id"]
            unique_to_gold[unique_id] = question
            unique_id += 1

            levels[data["level"]] += 1

        ### 2-hop
        else:
            fact_chain = [" ".join(ans_facts)] + [" ".join(other_facts)]

            entity_list_per_fact = []
            for fact in fact_chain:
                entity_list = []
                passed_entities = []
                for ent in nlp(fact).ents:
                    entity_list.append(ent.text)
                    if ent.label_ in ["PERSON", "FAC", "ORG", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW"]:
                        entity_list.append(ent.text)
                    else:
                        passed_entities.append(ent.text)
                for phrase in nlp(fact)._.phrases:
                    entity = phrase.text
                    if entity not in passed_entities:
                        entity_list.append(entity)
                entity_list_per_fact.append(list(set(entity_list)))

            total_input_ids = []
            total_attention_mask = []
            skip = False
            for i, fact in enumerate(fact_chain):
                if skip:
                    continue

                if i < len(fact_chain)-1:
                    bridges = set()
                    current_entities = [(set(entity.lower().split()), entity) for entity in entity_list_per_fact[i]]
                    next_entities = [(set(entity.lower().split()), entity) for entity in entity_list_per_fact[i+1]]

                    for cur_ent, entity in current_entities:
                        for next_ent, _ in next_entities:
                            if len(cur_ent&next_ent) > 0:
                                bridges.add(entity+".")
                    
                    if i == 0:
                        input_seq = "Answer: " + answer + " Fact: " + fact + " Bridges: " + " ".join(list(bridges))
                    else:
                        input_seq = "Fact: " + fact + " Bridges: " + " ".join(list(bridges))                 
                else:
                    input_seq = "Fact: " + fact

                encoded = tokenizer(input_seq)
                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]

                if len(input_ids) > args.max_encoder_length:
                    cut += 1
                    input_ids = input_ids[:args.max_encoder_length]
                    attention_mask = attention_mask[:args.max_encoder_length]

                while len(input_ids) < args.max_encoder_length:
                    input_ids.append(tokenizer.pad_token_id)
                    attention_mask.append(0)

                total_input_ids += input_ids
                total_attention_mask += attention_mask
            
            if skip:
                continue

            assert len(total_input_ids) == args.max_encoder_length * 2, f"{len(total_input_ids)}"
            assert len(total_attention_mask) == args.max_encoder_length * 2, f"{len(total_attention_mask)}"

            if mode == "train":
                output_seq = question
                encoded = tokenizer(output_seq)
                labels = encoded["input_ids"] + [-100]
                decoder_input_ids = [bos_token_id] + encoded["input_ids"]
                decoder_attention_mask = [1] + encoded["attention_mask"]

                if len(labels) > args.max_decoder_length:
                    print("Long Output:", len(labels))
                    continue

                while len(labels) < args.max_decoder_length:
                    labels.append(-100)
                    decoder_input_ids.append(tokenizer.pad_token_id)
                    decoder_attention_mask.append(0)

                assert len(decoder_input_ids) == args.max_decoder_length
                assert len(decoder_attention_mask) == args.max_decoder_length
                assert len(labels) == args.max_decoder_length
            else:
                labels = []
                decoder_input_ids = [bos_token_id]
                decoder_attention_mask = [1]

            feature = FeatInst(unique_id=unique_id,
                               input_ids=total_input_ids,
                               attention_mask=total_attention_mask,
                               decoder_input_ids=decoder_input_ids,
                               decoder_attention_mask=decoder_attention_mask,
                               labels=labels,
                               hop=2)
            features_per_hop[2].append(feature)
            unique_to_id[unique_id] = data["_id"]
            unique_to_gold[unique_id] = question
            unique_id += 1

            levels[data["level"]] += 1

    print("Truncated Input Num:", cut)
    print("Unintended Single Hop:", unintended_singleHop)
    print("Level:", levels)
    return features_per_hop, unique_to_id, unique_to_gold

def get_hotpot_features(data_file, tokenizer, mode):
    with open(data_file, "r") as fin:
        data_list = json.load(fin)

    bos_token_id = tokenizer.eos_token_id if "bart" in args.model_name.lower() else tokenizer.pad_token_id

    levels = {"easy": 0, "medium": 0, "hard": 0}
    features_per_hop = {1: [], 2: []}
    unique_to_id = dict()
    unique_to_gold = dict()
    unique_id = 0
    passed_input_num, passed_output_num = 0, 0
    unintended_singleHop = 0
    for data in tqdm.tqdm(data_list):
        title_to_lines = dict()
        for title, lines in data["context"]:
            title_to_lines[title] = lines

        facts = dict()
        try:
            for title, line_no in data["supporting_facts"]:
                if title in facts:
                    facts[title].append(title_to_lines[title][line_no])
                else:
                    facts[title] = [title_to_lines[title][line_no]]
        except:
            continue

        for title, lines in facts.items():
            facts[title] = "".join(lines)

        question = data["question"]
        answer = data["answer"]

        ans_facts = []
        other_facts = []
        for title, line in facts.items():
            sub_fact = title + ". " + line
            if answer in line:
                ans_facts.append(sub_fact)
            elif " ".join(answer.split()) in " ".join(line.split()):
                ans_facts.append(sub_fact)
            else:
                other_facts.append(sub_fact)

        if not ans_facts:
            if data["type"] == "comparison":
                ans_facts = other_facts
                other_facts = []
            else:
                print("Cannot find answer span")
                ans_facts = other_facts
                other_facts = []
                if mode == "train":
                    continue

        if len(other_facts) == 0:
            if data["level"] != "easy" and data["type"] == "bridge":
                unintended_singleHop += 1

            ans_fact = " ".join(ans_facts)

            input_seq = "Answer: " + answer + " Fact: " + ans_fact
            encoded_input = tokenizer(input_seq)
            input_ids = encoded_input["input_ids"]
            attention_mask = encoded_input["attention_mask"]

            if len(input_ids) > args.max_encoder_length:
                print(f"Long Input: {len(input_ids)}")
                input_ids = input_ids[:args.max_encoder_length]
                attention_mask = attention_mask[:args.max_encoder_length]

            while len(input_ids) < args.max_encoder_length:
                input_ids.append(tokenizer.pad_token_id)
                attention_mask.append(0)

            assert len(input_ids) == args.max_encoder_length
            assert len(attention_mask) == args.max_encoder_length

            if mode == "train":
                output_seq = question
                encoded_output = tokenizer(output_seq)
                labels = encoded_output["input_ids"] + [-100]
                decoder_input_ids = [bos_token_id] + encoded_output["input_ids"]
                decoder_attention_mask = [1] + encoded_output["attention_mask"]

                if len(labels) > args.max_decoder_length:
                    passed_output_num +=1
                    print(f"Long Output ({passed_output_num}): {len(labels)}")
                    continue

                while len(labels) < args.max_decoder_length:
                    labels.append(-100)
                    decoder_input_ids.append(tokenizer.pad_token_id)
                    decoder_attention_mask.append(0)
                
                assert len(decoder_input_ids) == args.max_decoder_length
                assert len(decoder_attention_mask) == args.max_decoder_length
                assert len(labels) == args.max_decoder_length
            else:
                labels = []
                decoder_input_ids = [bos_token_id]
                decoder_attention_mask = [1]

            feature = FeatInst(unique_id=unique_id,
                               input_ids=input_ids,
                               attention_mask=attention_mask,
                               decoder_input_ids=decoder_input_ids,
                               decoder_attention_mask=decoder_attention_mask,
                               labels=labels,
                               hop=1)
            features_per_hop[1].append(feature)
            unique_to_id[unique_id] = data["_id"]
            unique_to_gold[unique_id] = question
            unique_id += 1

            levels[data["level"]] += 1

        ### 2-hop
        else:
            fact_chain = [" ".join(ans_facts)] + [" ".join(other_facts)]

            entity_list_per_fact = []
            for fact in fact_chain:
                entity_list = []
                passed_entities = []
                for ent in nlp(fact).ents:
                    entity_list.append(ent.text)
                    if ent.label_ in ["PERSON", "FAC", "ORG", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW"]:
                        entity_list.append(ent.text)
                    else:
                        passed_entities.append(ent.text)
                for phrase in nlp(fact)._.phrases:
                    entity = phrase.text
                    if entity not in passed_entities:
                        entity_list.append(entity)
                entity_list_per_fact.append(list(set(entity_list)))

            total_input_ids = []
            total_attention_mask = []
            skip = False
            for i, fact in enumerate(fact_chain):
                if skip:
                    continue

                if i < len(fact_chain)-1:
                    bridges = set()
                    current_entities = [(set(entity.lower().split()), entity) for entity in entity_list_per_fact[i]]
                    next_entities = [(set(entity.lower().split()), entity) for entity in entity_list_per_fact[i+1]]

                    for cur_ent, entity in current_entities:
                        for next_ent, _ in next_entities:
                            if len(cur_ent&next_ent) > 0:
                                bridges.add(entity+".")
                    
                    if i == 0:
                        input_seq = "Answer: " + answer + " Fact: " + fact + " Bridges: " + " ".join(list(bridges))
                    else:
                        input_seq = "Fact: " + fact + " Bridges: " + " ".join(list(bridges))                 
                else:
                    input_seq = "Fact: " + fact

                encoded = tokenizer(input_seq)
                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]

                if len(input_ids) > args.max_encoder_length:
                    print(f"Long Input: {len(input_ids)}")
                    input_ids = input_ids[:args.max_encoder_length]
                    attention_mask = attention_mask[:args.max_encoder_length]

                while len(input_ids) < args.max_encoder_length:
                    input_ids.append(tokenizer.pad_token_id)
                    attention_mask.append(0)

                total_input_ids += input_ids
                total_attention_mask += attention_mask
            
            if skip:
                continue

            assert len(total_input_ids) == args.max_encoder_length * 2, f"{len(total_input_ids)}"
            assert len(total_attention_mask) == args.max_encoder_length * 2, f"{len(total_attention_mask)}"

            if mode == "train":
                output_seq = question
                encoded = tokenizer(output_seq)
                labels = encoded["input_ids"] + [-100]
                decoder_input_ids = [bos_token_id] + encoded["input_ids"]
                decoder_attention_mask = [1] + encoded["attention_mask"]

                if len(labels) > args.max_decoder_length:
                    print("Long Output:", len(labels))
                    continue

                while len(labels) < args.max_decoder_length:
                    labels.append(-100)
                    decoder_input_ids.append(tokenizer.pad_token_id)
                    decoder_attention_mask.append(0)

                assert len(decoder_input_ids) == args.max_decoder_length
                assert len(decoder_attention_mask) == args.max_decoder_length
                assert len(labels) == args.max_decoder_length
            else:
                labels = []
                decoder_input_ids = [bos_token_id]
                decoder_attention_mask = [1]

            feature = FeatInst(unique_id=unique_id,
                               input_ids=total_input_ids,
                               attention_mask=total_attention_mask,
                               decoder_input_ids=decoder_input_ids,
                               decoder_attention_mask=decoder_attention_mask,
                               labels=labels,
                               hop=2)
            features_per_hop[2].append(feature)
            unique_to_id[unique_id] = data["_id"]
            unique_to_gold[unique_id] = question
            unique_id += 1

            levels[data["level"]] += 1

    print("Passed Num:", passed_input_num, passed_output_num)
    print("Unintended Single Hop:", unintended_singleHop)
    print("Level:", levels)
    return features_per_hop, unique_to_id, unique_to_gold

class GDataset(Dataset):
    def __init__(self, data_file, tokenizer, device, mode):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device

        print(f"Generate features...")
        if "hotpot" in data_file.lower():
            if args.paralevel:
                print("Paragraph-level")
                self.features_per_hop, self.unique_to_id, self.unique_to_gold = get_hotpot_features_paralevel(data_file, tokenizer, mode)
            else:
                print("Sentence-level")
                self.features_per_hop, self.unique_to_id, self.unique_to_gold = get_hotpot_features(data_file, tokenizer, mode)
        elif "musique" in data_file.lower():
            self.features_per_hop, self.unique_to_id, self.unique_to_gold = get_musique_features(data_file, tokenizer, mode)
        else:
            assert False
        self.features = []
        self.hops = sorted(list(self.features_per_hop.keys()))
        self.max_features_num = 0
        for feats in self.features_per_hop.values():
            self.features += feats
            if self.max_features_num < len(feats):
                self.max_features_num = len(feats)
        self.mix_batch = False

    def set_hop(self, hop):
        if hop not in self.features_per_hop:
            print("hop out-of-range:", self.hops)
            assert False
        self.features = self.features_per_hop[hop]

    def do_mix_batch(self):
        self.mix_batch = True

    def __len__(self):
        #if self.mix_batch:
        #    return self.max_features_num
        return len(self.features)
    
    def __getitem__(self, idx):
        #if idx >= len(self.features):
        #    idx = idx % len(self.features)
        return self.features[idx]
    
    def collate_fn(self, batch):
        for i, feature in enumerate(batch):
            batch[i] = FeatInst(unique_id=np.asarray(feature.unique_id),
                                input_ids=np.asarray(feature.input_ids),
                                attention_mask=np.asarray(feature.attention_mask),
                                decoder_input_ids=np.asarray(feature.decoder_input_ids),
                                decoder_attention_mask=np.asarray(feature.decoder_attention_mask),
                                labels=np.asarray(feature.labels),
                                hop=np.asarray(feature.hop))
        results = FeatInst(*(default_collate(samples) for samples in zip(*batch)))
        return results
    
class Model(nn.Module):
    def __init__(self, model_name, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.EncoderDecoder = ConditionalGeneration.from_pretrained(model_name)

    def forward(self, hop, total_input_ids, total_attention_mask, decoder_input_ids, decoder_attention_mask, labels):
        batch_size = total_input_ids.size(0)
        input_ids_per_hop = total_input_ids.reshape((batch_size, hop, -1)).transpose(0, 1)
        attention_mask_per_hop = total_attention_mask.reshape((batch_size, hop, -1)).transpose(0, 1)

        past_key_values = None
        accum_decoder_attention_mask = []
        accum_cross_attention_mask = []
        ## intermediate
        for i in range(hop-1):
            accum_cross_attention_mask.append(attention_mask_per_hop[i])
            OutputDict = self.EncoderDecoder.intermediate_forward(input_ids=input_ids_per_hop[i],
                                                                  attention_mask=attention_mask_per_hop[i],
                                                                  cross_attention_mask=torch.cat(accum_cross_attention_mask, dim=-1),
                                                                  decoder_attention_mask=torch.cat(accum_decoder_attention_mask, dim=-1) if accum_decoder_attention_mask else None,
                                                                  past_key_values=past_key_values,
                                                                  max_length=args.max_decoder_length,
                                                                  return_dict_in_generate=True)
            past_key_values = OutputDict["past_key_values"] # [layer, 2*[(batch, num_head, decoder_length, emb_size) + 2*[(batch, num_head, encoder_length, emb_size)]]]
            accum_decoder_attention_mask.append(OutputDict["decoder_attention_mask"])
#
        ## Last
        accum_cross_attention_mask.append(attention_mask_per_hop[-1])
        accum_decoder_attention_mask.append(decoder_attention_mask)
        OutputDict = self.EncoderDecoder(input_ids=input_ids_per_hop[-1],
                                         attention_mask=attention_mask_per_hop[-1],
                                         decoder_input_ids=decoder_input_ids,
                                         cross_attention_mask=torch.cat(accum_cross_attention_mask, dim=-1),
                                         decoder_attention_mask=torch.cat(accum_decoder_attention_mask, dim=-1),
                                         labels=labels,
                                         past_key_values=past_key_values,
                                         return_dict=True)
        
        return OutputDict.loss
    
    def generate(self, hop, total_input_ids, total_attention_mask):
        bos_token_id = self.tokenizer.eos_token_id if "bart" in args.model_name.lower() else self.tokenizer.pad_token_id ## t5
        device = total_input_ids.device
        batch_size = total_input_ids.size(0)
        input_ids_per_hop = total_input_ids.reshape((batch_size, hop, -1)).transpose(0, 1)
        attention_mask_per_hop = total_attention_mask.reshape((batch_size, hop, -1)).transpose(0, 1)

        past_key_values = None
        accum_decoder_attention_mask = []
        accum_cross_attention_mask = []
        ## intermediate
        for i in range(hop-1):
            accum_cross_attention_mask.append(attention_mask_per_hop[i])
            OutputDict = self.EncoderDecoder.intermediate_forward(input_ids=input_ids_per_hop[i],
                                                                  attention_mask=attention_mask_per_hop[i],
                                                                  cross_attention_mask=torch.cat(accum_cross_attention_mask, dim=-1),
                                                                  decoder_attention_mask=torch.cat(accum_decoder_attention_mask, dim=-1) if accum_decoder_attention_mask else None,
                                                                  past_key_values=past_key_values,
                                                                  max_length=args.max_decoder_length,
                                                                  return_dict_in_generate=True)
            past_key_values = OutputDict["past_key_values"]
            accum_decoder_attention_mask.append(OutputDict["decoder_attention_mask"])

        # Last
        accum_cross_attention_mask.append(attention_mask_per_hop[-1])
        output = self.EncoderDecoder.generate(input_ids=input_ids_per_hop[-1],
                                              attention_mask=attention_mask_per_hop[-1],
                                              cross_attention_mask=torch.cat(accum_cross_attention_mask, dim=-1),
                                              decoder_input_ids=torch.tensor([[bos_token_id]]*batch_size).to(device),
                                              decoder_attention_mask=torch.cat(accum_decoder_attention_mask, dim=-1) if accum_decoder_attention_mask else None,
                                              past_key_values=past_key_values,
                                              num_beams=1,
                                              max_length=args.max_decoder_length)
        preds = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        return preds
    
    def generate_step_by_step(self, hop, total_input_ids, total_attention_mask):
        bos_token_id = self.tokenizer.eos_token_id if "bart" in args.model_name.lower() else self.tokenizer.pad_token_id ## t5
        device = total_input_ids.device
        batch_size = total_input_ids.size(0)
        input_ids_per_hop = total_input_ids.reshape((batch_size, hop, -1)).transpose(0, 1)
        attention_mask_per_hop = total_attention_mask.reshape((batch_size, hop, -1)).transpose(0, 1)

        prediction_per_hop = dict()

        past_key_values = None
        accum_decoder_attention_mask = []
        accum_cross_attention_mask = []
        ## intermediate
        for i in range(hop-1):
            accum_cross_attention_mask.append(attention_mask_per_hop[i])
            OutputDict = self.EncoderDecoder.intermediate_forward(input_ids=input_ids_per_hop[i],
                                                                  attention_mask=attention_mask_per_hop[i],
                                                                  cross_attention_mask=torch.cat(accum_cross_attention_mask, dim=-1),
                                                                  decoder_attention_mask=torch.cat(accum_decoder_attention_mask, dim=-1) if accum_decoder_attention_mask else None,
                                                                  past_key_values=past_key_values,
                                                                  max_length=args.max_decoder_length,
                                                                  return_dict_in_generate=True)
            past_key_values = OutputDict["past_key_values"]
            accum_decoder_attention_mask.append(OutputDict["decoder_attention_mask"])
            prediction_per_hop[i+1] = self.tokenizer.batch_decode(OutputDict["predictions"], skip_special_tokens=True)

        # Last
        accum_cross_attention_mask.append(attention_mask_per_hop[-1])
        output = self.EncoderDecoder.generate(input_ids=input_ids_per_hop[-1],
                                              attention_mask=attention_mask_per_hop[-1],
                                              cross_attention_mask=torch.cat(accum_cross_attention_mask, dim=-1),
                                              decoder_input_ids=torch.tensor([[bos_token_id]]*batch_size).to(device),
                                              decoder_attention_mask=torch.cat(accum_decoder_attention_mask, dim=-1) if accum_decoder_attention_mask else None,
                                              past_key_values=past_key_values,
                                              num_beams=1,
                                              max_length=args.max_decoder_length)
        preds = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        prediction_per_hop[hop] = preds

        return prediction_per_hop
    
class DoubleModel(nn.Module):
    def __init__(self, model_name, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.EncoderDecoder = ConditionalGeneration.from_pretrained(model_name)

    def forward(self, hop, total_input_ids, total_attention_mask, decoder_input_ids, decoder_attention_mask, labels):
        batch_size = total_input_ids.size(0)
        input_ids_per_hop = total_input_ids.reshape((batch_size, hop, -1)).transpose(0, 1)
        attention_mask_per_hop = total_attention_mask.reshape((batch_size, hop, -1)).transpose(0, 1)

        encoder_past_key_values = None
        past_key_values = None
        accum_decoder_attention_mask = []
        accum_cross_attention_mask = []
        ## intermediate
        for i in range(hop-1):
            accum_cross_attention_mask.append(attention_mask_per_hop[i])
            OutputDict = self.EncoderDecoder.intermediate_forward(input_ids=input_ids_per_hop[i],
                                                                  attention_mask=torch.cat(accum_cross_attention_mask, dim=-1),
                                                                  cross_attention_mask=torch.cat(accum_cross_attention_mask, dim=-1),
                                                                  decoder_attention_mask=torch.cat(accum_decoder_attention_mask, dim=-1) if accum_decoder_attention_mask else None,
                                                                  past_key_values=past_key_values,
                                                                  encoder_past_key_values=encoder_past_key_values,
                                                                  max_length=args.max_decoder_length,
                                                                  return_dict_in_generate=True)
            encoder_past_key_values = OutputDict["encoder_past_key_values"]
            past_key_values = OutputDict["past_key_values"] # [layer, 2*[(batch, num_head, decoder_length, emb_size) + 2*[(batch, num_head, encoder_length, emb_size)]]]
            accum_decoder_attention_mask.append(OutputDict["decoder_attention_mask"])
#
        ## Last
        accum_cross_attention_mask.append(attention_mask_per_hop[-1])
        accum_decoder_attention_mask.append(decoder_attention_mask)
        OutputDict = self.EncoderDecoder(input_ids=input_ids_per_hop[-1],
                                         attention_mask=torch.cat(accum_cross_attention_mask, dim=-1),
                                         decoder_input_ids=decoder_input_ids,
                                         cross_attention_mask=torch.cat(accum_cross_attention_mask, dim=-1),
                                         decoder_attention_mask=torch.cat(accum_decoder_attention_mask, dim=-1),
                                         labels=labels,
                                         past_key_values=past_key_values,
                                         encoder_past_key_values=encoder_past_key_values,
                                         return_dict=True)
        
        return OutputDict.loss
    
    def generate(self, hop, total_input_ids, total_attention_mask):
        bos_token_id = self.tokenizer.eos_token_id if "bart" in args.model_name.lower() else self.tokenizer.pad_token_id ## t5
        device = total_input_ids.device
        batch_size = total_input_ids.size(0)
        input_ids_per_hop = total_input_ids.reshape((batch_size, hop, -1)).transpose(0, 1)
        attention_mask_per_hop = total_attention_mask.reshape((batch_size, hop, -1)).transpose(0, 1)

        encoder_past_key_values = None
        past_key_values = None
        accum_decoder_attention_mask = []
        accum_cross_attention_mask = []
        ## intermediate
        for i in range(hop-1):
            accum_cross_attention_mask.append(attention_mask_per_hop[i])
            OutputDict = self.EncoderDecoder.intermediate_forward(input_ids=input_ids_per_hop[i],
                                                                  attention_mask=torch.cat(accum_cross_attention_mask, dim=-1),
                                                                  cross_attention_mask=torch.cat(accum_cross_attention_mask, dim=-1),
                                                                  decoder_attention_mask=torch.cat(accum_decoder_attention_mask, dim=-1) if accum_decoder_attention_mask else None,
                                                                  past_key_values=past_key_values,
                                                                  encoder_past_key_values=encoder_past_key_values,
                                                                  max_length=args.max_decoder_length,
                                                                  return_dict_in_generate=True)
            encoder_past_key_values = OutputDict["encoder_past_key_values"]
            past_key_values = OutputDict["past_key_values"]
            accum_decoder_attention_mask.append(OutputDict["decoder_attention_mask"])

        # Last
        accum_cross_attention_mask.append(attention_mask_per_hop[-1])
        output = self.EncoderDecoder.generate(input_ids=input_ids_per_hop[-1],
                                              attention_mask=torch.cat(accum_cross_attention_mask, dim=-1),
                                              cross_attention_mask=torch.cat(accum_cross_attention_mask, dim=-1),
                                              decoder_input_ids=torch.tensor([[bos_token_id]]*batch_size).to(device),
                                              decoder_attention_mask=torch.cat(accum_decoder_attention_mask, dim=-1) if accum_decoder_attention_mask else None,
                                              past_key_values=past_key_values,
                                              encoder_past_key_values=encoder_past_key_values,
                                              num_beams=1,
                                              max_length=args.max_decoder_length)
        preds = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        return preds
    
    def generate_step_by_step(self, hop, total_input_ids, total_attention_mask):
        bos_token_id = self.tokenizer.eos_token_id if "bart" in args.model_name.lower() else self.tokenizer.pad_token_id ## t5
        device = total_input_ids.device
        batch_size = total_input_ids.size(0)
        input_ids_per_hop = total_input_ids.reshape((batch_size, hop, -1)).transpose(0, 1)
        attention_mask_per_hop = total_attention_mask.reshape((batch_size, hop, -1)).transpose(0, 1)

        prediction_per_hop = dict()

        encoder_past_key_values = None
        past_key_values = None
        accum_decoder_attention_mask = []
        accum_cross_attention_mask = []
        ## intermediate
        for i in range(hop-1):
            accum_cross_attention_mask.append(attention_mask_per_hop[i])
            OutputDict = self.EncoderDecoder.intermediate_forward(input_ids=input_ids_per_hop[i],
                                                                  attention_mask=torch.cat(accum_cross_attention_mask, dim=-1),
                                                                  cross_attention_mask=torch.cat(accum_cross_attention_mask, dim=-1),
                                                                  decoder_attention_mask=torch.cat(accum_decoder_attention_mask, dim=-1) if accum_decoder_attention_mask else None,
                                                                  past_key_values=past_key_values,
                                                                  encoder_past_key_values=encoder_past_key_values,
                                                                  max_length=args.max_decoder_length,
                                                                  return_dict_in_generate=True)
            encoder_past_key_values = OutputDict["encoder_past_key_values"]
            past_key_values = OutputDict["past_key_values"]
            accum_decoder_attention_mask.append(OutputDict["decoder_attention_mask"])
            prediction_per_hop[i+1] = self.tokenizer.batch_decode(OutputDict["predictions"], skip_special_tokens=True)

        # Last
        accum_cross_attention_mask.append(attention_mask_per_hop[-1])
        output = self.EncoderDecoder.generate(input_ids=input_ids_per_hop[-1],
                                              attention_mask=torch.cat(accum_cross_attention_mask, dim=-1),
                                              cross_attention_mask=torch.cat(accum_cross_attention_mask, dim=-1),
                                              decoder_input_ids=torch.tensor([[bos_token_id]]*batch_size).to(device),
                                              decoder_attention_mask=torch.cat(accum_decoder_attention_mask, dim=-1) if accum_decoder_attention_mask else None,
                                              past_key_values=past_key_values,
                                              encoder_past_key_values=encoder_past_key_values,
                                              num_beams=1,
                                              max_length=args.max_decoder_length)
        preds = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        prediction_per_hop[hop] = preds

        return prediction_per_hop
    
def train(model, train_dataset, valid_dataset, model_dir, device, start_epoch=0, optimizer_state=None, scheduler_state=None, opt_checkpoint=None, max_score=-1):
    def get_loaders(loaders, hops, epoch):
        if len(hops) == 1:
            total_loader = tqdm.tqdm(zip(loaders[1]), total=len(loaders[1]), desc=f"Train Epoch-{epoch} | MIX BATCH {hops}")
        elif len(hops) == 2:
            total_loader = tqdm.tqdm(zip(loaders[1], loaders[2]), total=min([len(loaders[i]) for i in hops]), desc=f"Train Epoch-{epoch} | MIX BATCH {hops}")
        elif len(hops) == 3:
            total_loader = tqdm.tqdm(zip(loaders[1], loaders[2], loaders[3]), total=min([len(loaders[i]) for i in hops]), desc=f"Train Epoch-{epoch} | MIX BATCH {hops}")
        elif len(hops) == 4:
            total_loader = tqdm.tqdm(zip(loaders[1], loaders[2], loaders[3], loaders[4]), total=min([len(loaders[i]) for i in hops]), desc=f"Train Epoch-{epoch} | MIX BATCH {hops}")
        return total_loader

    progress_bar = tqdm.tqdm
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
        
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
    num_training_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, num_training_steps)
    if scheduler_state:
        scheduler.load_state_dict(scheduler_state)

    hops = train_dataset.hops
    num_hold = args.early_stop
    total_train_losses = []
    total_loader = None
    for epoch in range(start_epoch, args.epochs):
        epoch = epoch + 1
        train_loss = []
        optimizer.zero_grad()
        used_hops = None

        if args.mix_batch:
            model.train()
            model.zero_grad()
            
            loaders = dict()
            total_batch = 0
            for hop in train_dataset.hops:
                if args.start_epochs[hop] > epoch:
                    break
                new_dataset = copy.deepcopy(train_dataset)
                new_dataset.do_mix_batch()
                new_dataset.set_hop(hop)
                train_loader = DataLoader(new_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=new_dataset.collate_fn)
                total_batch += len(new_dataset)
                del new_dataset
                loaders[hop] = train_loader
            used_hops = list(loaders.keys())

            total_loader = get_loaders(loaders, used_hops, epoch)
            for batch_list in total_loader:
                local_losses = torch.tensor(0.0).to(device)
                for batch in batch_list:
                    hop = batch.hop[0].item()
                    loss = model(hop, 
                                batch.input_ids.to(device), 
                                batch.attention_mask.to(device),
                                batch.decoder_input_ids.to(device),
                                batch.decoder_attention_mask.to(device),
                                batch.labels.to(device))
                    train_loss.append(float(loss))
                    local_losses += loss

                local_losses.backward()
                total_train_losses.append(float(local_losses)/len(used_hops))

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()
                    
                total_loader.set_postfix(loss=float(local_losses)/len(used_hops))
            print("TRAIN LOSS:", round(sum(train_loss)/len(train_loss), 5))

        else:
            model.train()
            model.zero_grad()
            used_hops = []
            for hop in hops:
                if args.start_epochs[hop] > epoch or args.end_epochs[hop] < epoch:
                    continue
                used_hops.append(hop)
                train_dataset.set_hop(hop)
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
                train_pbar = progress_bar(train_loader, total=len(train_loader), desc=f"Train Epoch-{epoch} | Hop-{hop}")
                
                for batch in train_pbar:
                    loss = model(hop, 
                                batch.input_ids.to(device), 
                                batch.attention_mask.to(device),
                                batch.decoder_input_ids.to(device),
                                batch.decoder_attention_mask.to(device),
                                batch.labels.to(device))
                    train_loss.append(float(loss))
                    total_train_losses.append(float(loss))
                    loss.backward()
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    model.zero_grad()
                    
                    train_pbar.set_postfix(loss=float(loss))
                print("TRAIN LOSS:", round(sum(train_loss)/len(train_loss), 5))

        with torch.no_grad():
            valid_score = predict(model, valid_dataset, device, desc=f"Valid Epoch-{epoch}", hops=used_hops)

        checkpoint = f"{epoch}_{valid_score}.pth"
        print(f"Save the model to {os.path.join(model_dir, checkpoint)}")
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'opt_checkpoint': opt_checkpoint,
                    'max_score': max_score}, os.path.join(model_dir, checkpoint))

        if len(hops) == len(used_hops):
            if max_score < valid_score:
                max_score = valid_score
                opt_checkpoint = checkpoint
                num_hold = args.early_stop
            else:
                num_hold -= 1

        plt.plot(total_train_losses)
        plt.savefig(os.path.join(model_dir, f"train-loss.png"))
        plt.cla()

        if num_hold == 0:
            print("Early Stop")
            break

        del total_loader
        torch.cuda.empty_cache()

    return os.path.join(model_dir, opt_checkpoint)

def get_scores(predictions, references):
    preds = []
    golds = []
    m_score = 0.0
    rl_score = 0.0
    rouge = Rouge()
    num = 0

    try:    
        for pred, gold in zip(predictions, references):
            rl_score += rouge.get_scores([pred], [gold])[0]['rouge-l']['f']
            pred = nltk.word_tokenize(pred)
            gold = nltk.word_tokenize(gold)
            preds.append(pred)
            golds.append([gold])
            num += 1
            m_score += nltk.translate.meteor_score.meteor_score([gold], pred)
        bleu_score = nltk.translate.bleu_score.corpus_bleu(golds, preds) * 100
        m_score /= num
        m_score *= 100
        rl_score /= num
        rl_score *= 100
    except:
        return None

    return bleu_score, m_score, rl_score

def predict(model, dataset, device, desc="Test", result_file=None, hops=None):
    model.eval()
    if hops is None:
        hops = dataset.hops
    results = dict()
    total_preds = []
    total_refs = []
    performances_per_hop = dict()
    for hop in hops:
        id_list = []
        predictions = []
        references = []

        dataset.set_hop(hop)
        dataloader = DataLoader(dataset, batch_size=args.valid_batch_size, shuffle=False, collate_fn=dataset.collate_fn)
        pbar = tqdm.tqdm(dataloader, total=len(dataloader), desc=f"{desc} | Hop-{hop}")
        for batch in pbar:
            preds = model.generate(hop,
                                   total_input_ids=batch.input_ids.to(device), 
                                   total_attention_mask=batch.attention_mask.to(device))
            
            predictions.extend(preds)

            for unique_id in batch.unique_id:
                unique_id = unique_id.item()
                id_list.append(dataset.unique_to_id[unique_id])
                references.append(dataset.unique_to_gold[unique_id])

        if len(predictions) == 0:
            continue

        score = get_scores(predictions, references)
        if score is None:
            print("PASS EVALUATION")
        else:
            bleu, meteor, rougel = score
            print("BLEU:", bleu)
            print("METEOR:", meteor)
            print("ROUGE-L:", rougel)
            performances_per_hop[hop] = {"bleu": bleu,
                                        "meteor": meteor,
                                        "rougel": rougel}
            
            for _id, pred, gold in zip(id_list, predictions, references):
                results[_id] = {"pred": pred, "gold": gold, "hop": hop}

            total_preds += predictions
            total_refs += references

        del dataloader
        torch.cuda.empty_cache()

    total_performances = {"bleu": 0, "meteor": 0, "rougel": 0}
    for metric in total_performances.keys():
        for performances in performances_per_hop.values():
            total_performances[metric] += performances[metric]
        total_performances[metric] /= len(performances_per_hop.keys())

    performances_per_hop["total"] = total_performances
    
    print("### Total Performances ###")
    print("BLEU:", bleu)
    print("METEOR:", meteor)
    print("ROUGE-L:", rougel)

    if result_file:
        with open(result_file, "w") as fout:
            json.dump({"performance": performances_per_hop, "results": results}, fout, indent=1)

    return (bleu + meteor + rougel)/3

def analyze(model, dataset, device, result_file=None, hops=None):
    assert result_file is not None

    model.eval()
    if hops is None:
        hops = dataset.hops
    results = dict()

    for hop in hops:
        dataset.set_hop(hop)
        dataloader = DataLoader(dataset, batch_size=args.valid_batch_size, shuffle=False, collate_fn=dataset.collate_fn)
        pbar = tqdm.tqdm(dataloader, total=len(dataloader), desc=f"Analyze | Hop-{hop}")
        for batch in pbar:
            prediction_per_hop = model.generate_step_by_step(hop,
                                                             total_input_ids=batch.input_ids.to(device), 
                                                             total_attention_mask=batch.attention_mask.to(device))

            for i, unique_id in enumerate(batch.unique_id):
                unique_id = unique_id.item()
                _id = dataset.unique_to_id[unique_id]
                intermediates = dict()
                for h, preds in prediction_per_hop.items():
                    intermediates[h] = preds[i]
                results[_id] = {"pred": intermediates,
                                "gold": dataset.unique_to_gold[unique_id]}

    with open(result_file, "w") as fout:
        json.dump(results, fout, indent=1)
    
if __name__ == "__main__":
    parse_argument()
    seed_everything(args.seed)
    device = torch.device("cuda")

    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, args.exp_tag)
    os.makedirs(model_dir, exist_ok=True)
    config = os.path.join(model_dir, "config.json")

    if args.checkpoint:
        with open(config, "r") as fin:
            arg_dict = json.load(fin)
            args.seed = arg_dict["seed"]
            args.train_data_file = arg_dict["train_data_file"]
            args.valid_data_file = arg_dict["valid_data_file"]
            args.test_data_file = arg_dict["test_data_file"]
            args.model_name = arg_dict["model_name"]
            args.max_encoder_length = arg_dict["max_encoder_length"]
            args.max_decoder_length = arg_dict["max_decoder_length"]
            args.mix_batch = arg_dict["mix_batch"]
            args.last_mix_batch = arg_dict["last_mix_batch"]
            args.double = arg_dict["double"]
            if "paralevel" in arg_dict:
                args.paralevel = arg_dict["paralevel"]
            else:
                args.paralevel = False
            start_epochs = arg_dict["start_epochs"]
            for k, v in start_epochs.items():
                args.start_epochs[int(k)] = int(v)
            end_epochs = arg_dict["end_epochs"]
            for k, v in end_epochs.items():
                args.end_epochs[int(k)] = int(v)
    else:
        with open(config, "w") as fout:
            json.dump(vars(args), fout, indent=1)
    print(args)

    if "t5" in args.model_name:
        if args.double:
            from models_all import T5ForConditionalGeneration as ConditionalGeneration
        else:
            from models import T5ForConditionalGeneration as ConditionalGeneration

    else:
        if args.double:
            from models_all import BartForConditionalGeneration as ConditionalGeneration
        else:
            from models import BartForConditionalGeneration as ConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank")

    opt_checkpoint = None
    if args.do_train:
        print("Generate valid features...")
        valid_dataset = GDataset(args.valid_data_file, tokenizer, device, mode="valid")

        print("Generate train features...")
        train_dataset = GDataset(args.train_data_file, tokenizer, device, mode="train")

        if args.double:
            model = DoubleModel(args.model_name, tokenizer).cpu()
        else:
            model = Model(args.model_name, tokenizer).cpu()

        if args.checkpoint:
            print("Use checkpoint:", args.checkpoint)
            checkpoint_dict = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint_dict["model_state_dict"])
            model = model.to(device)

            opt_checkpoint = train(model, train_dataset, valid_dataset, model_dir, device, start_epoch=checkpoint_dict["epoch"], 
                  optimizer_state=checkpoint_dict["optimizer_state_dict"], scheduler_state=checkpoint_dict["scheduler_state_dict"], 
                  opt_checkpoint=checkpoint_dict["opt_checkpoint"], max_score=checkpoint_dict["max_score"])
        else:
            model = model.to(device)
            opt_checkpoint = train(model, train_dataset, valid_dataset, model_dir, device)

    if args.do_eval:
        print("Generate valid features...")
        valid_dataset = GDataset(args.valid_data_file, tokenizer, device, mode="eval")

        if args.double:
            model = DoubleModel(args.model_name, tokenizer).cpu()
        else:
            model = Model(args.model_name, tokenizer).cpu()

        if opt_checkpoint:
            model.load_state_dict(torch.load(opt_checkpoint)["model_state_dict"])
            checkpoint = opt_checkpoint
        elif args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint)["model_state_dict"])
            checkpoint = args.checkpoint
        else:
            assert False, "checkpoint required"
        model = model.to(device)

        result_file = checkpoint + "-Eval_results.json"
        predict(model, valid_dataset, device, desc="Evaluation", result_file=result_file)

    if args.do_test:
        print("Generate test features...")
        test_dataset = GDataset(args.test_data_file, tokenizer, device, mode="test")

        if args.double:
            model = DoubleModel(args.model_name, tokenizer).cpu()
        else:
            model = Model(args.model_name, tokenizer).cpu()

        if opt_checkpoint:
            model.load_state_dict(torch.load(opt_checkpoint)["model_state_dict"])
            checkpoint = opt_checkpoint
        elif args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint)["model_state_dict"])
            checkpoint = args.checkpoint
        else:
            assert False, "checkpoint required"
        model = model.to(device)

        result_file = checkpoint + "-Test_results.json"
        predict(model, test_dataset, device, desc="Test", result_file=result_file)

    if args.do_analyze:
        print("Generate test features...")
        test_dataset = GDataset(args.test_data_file, tokenizer, device, mode="test")

        if args.double:
            model = DoubleModel(args.model_name, tokenizer).cpu()
        else:
            model = Model(args.model_name, tokenizer).cpu()
            
        if opt_checkpoint:
            model.load_state_dict(torch.load(opt_checkpoint)["model_state_dict"])
            checkpoint = opt_checkpoint
        elif args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint)["model_state_dict"])
            checkpoint = args.checkpoint
        else:
            assert False, "checkpoint required"
        model = model.to(device)

        result_file = checkpoint + "-analysis.json"
        analyze(model, test_dataset, device, result_file=result_file, hops=[2,3,4])