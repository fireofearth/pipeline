import argparse
import os
import json
import h5py
import numpy
import cv2 as cv
from patch_dataset import PatchDataset
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pynvml import *
import aim_logger
import aim_models
import time
import sys
import re
import sklearn
from sklearn import metrics
import math

# Folder permission mode
p_mode = 0o777
oldmask = os.umask(000)

nvmlInit()

def build_model(model_config):

    model = aim_models.DeepModel(model_config)

    return model

def get_auc_value(pred_labels_probs, batch_labels):

    # overall_auc_roc_score = sklearn.metrics.roc_auc_score(
    #         batch_labels, 
    #         numpy.argmax(pred_labels_probs, axis=1),
    #         average='macro')
    print('type(pred_labels_probs)', type(pred_labels_probs))
    print('pred_labels_probs.shape', pred_labels_probs.shape)
    print('type(batch_labels)', type(batch_labels))
    print('batch_labels.shape', batch_labels.shape)
    pred_labels_probs = numpy.squeeze(numpy.array(pred_labels_probs), 0)
    batch_labels = numpy.squeeze(numpy.array(batch_labels), 0)
    overall_auc_roc_score = sklearn.metrics.roc_auc_score(batch_labels, pred_labels_probs, average='macro', multi_class='ovo')

    return overall_auc_roc_score


def get_kappa_value(pred_labels_probs, batch_labels):

    _, predicted = torch.max(pred_labels_probs, 1)
    total_num_interval_patches = batch_labels.size(0)

    kappa = sklearn.metrics.cohen_kappa_score(predicted.cpu(), batch_labels.cpu())

    return kappa


def get_class_accuracy_sets(pred_labels_probs, batch_labels):

    type_dict = {}

    _, predicted = torch.max(pred_labels_probs, 1)
    predicted = predicted.cpu()
    batch_labels_cpu = batch_labels.cpu()

    for i in range(len(batch_labels_cpu)):
        type_dict[batch_labels_cpu[i].item()] = [0, 0]

    for i in range(len(batch_labels_cpu)):           
        type_dict[batch_labels_cpu[i].item()][0] += 1

        if predicted[i] == batch_labels_cpu[i]:
            type_dict[batch_labels_cpu[i].item()][1] += 1

    return type_dict


def get_accuracy_sets(pred_labels_probs, batch_labels):

    _, predicted = torch.max(pred_labels_probs, 1)
    total_num_interval_patches = batch_labels.size(0)
    total_num_correct_interval_patches = (predicted == batch_labels).sum().item()

    return { "total" : total_num_interval_patches, "correct" : total_num_correct_interval_patches}

def test(model, test_loader, path_value_to_index):
    model.model.eval()

    index_to_subtype = {}

    for key, value in path_value_to_index.items():
        index_to_subtype[value] = key

    correct = 0
    total = 0
    loss = 0

    #kappa_values = []
    auc_values = []
    class_accuracy_values = []

    all_pred_labels_probs = []
    all_batch_labels = []

    all_probs_tensor = None
    all_batch_labels_tensor = None

    preds = numpy.array([]).reshape(0, len(index_to_subtype.keys()))
    labels = []

    with torch.no_grad():
        for batch_index, data in enumerate(test_loader):

            batch_data, batch_labels = data
            batch_data = batch_data.cuda()
            batch_labels = batch_labels.cuda()

            pred_labels_logits, pred_labels_probs, output = model.forward(batch_data)
            loss += model.get_loss(pred_labels_logits, batch_labels, output)

            accuracy_sets = get_accuracy_sets(pred_labels_probs, batch_labels)
            class_accuracy_sets = get_class_accuracy_sets(pred_labels_probs, batch_labels)
            class_accuracy_values.append(class_accuracy_sets)

            kappa = get_kappa_value(pred_labels_probs, batch_labels)

            all_pred_labels_probs.extend(pred_labels_probs)
            all_batch_labels.extend(batch_labels)

            if all_probs_tensor == None:
                all_probs_tensor = pred_labels_probs
            else:
                all_probs_tensor = torch.cat([pred_labels_probs, all_probs_tensor], dim=0)

            if all_batch_labels_tensor == None:
                all_batch_labels_tensor = batch_labels
            else:
                all_batch_labels_tensor = torch.cat([batch_labels, all_batch_labels_tensor], dim=0)

            #if not math.isnan(kappa):
            #    kappa_values.append(kappa)

            preds = numpy.vstack((preds, pred_labels_probs.cpu().numpy()))
            labels.extend(batch_labels.cpu().numpy().tolist())

            total += accuracy_sets["total"]
            correct += accuracy_sets["correct"]
            print("T A:{} L:{} K:{}".format(correct / total, loss, get_kappa_value(all_probs_tensor, all_batch_labels_tensor)))

    auc = get_auc_value(preds, labels)
    acc = correct / total
    kappa = get_kappa_value(all_probs_tensor, all_batch_labels_tensor)
    print("Final T A:{} L:{} K:{} AUC:{}".format(acc, loss, kappa, auc))

    class_acc = {}

    for acc_itr in class_accuracy_values:
        for key, value in acc_itr.items():

            if key not in class_acc:
                class_acc[key] = []

            class_acc[key].append(value)

    print("Class Acc %")
    class_acc_percent = {}
    for key, value in class_acc.items():
        total = 0
        correct = 0

        for i in value:
            total += i[0]
            correct += i[1]

        class_acc_percent[key] = correct / total
        print("{}: {}".format(index_to_subtype[key], correct / total))

    model.model.train()

    return {"acc" : float(acc), "loss" : float(loss.cpu().item()), "kappa" : kappa, "auc" : auc, "class_acc" : class_acc_percent} 

def load_chunk_file(chunk_file_path, chunks_to_use):

    chunks_to_use = [int(i) for i in str(chunks_to_use).split("_")]

    img_paths = []
    
    with open(chunk_file_path) as json_file:
        data = json.load(json_file)

        for chunk in data['chunks']:
            if chunk['id'] in chunks_to_use:
                img_paths.extend(chunk['imgs'])

    return img_paths

def generate_x_y_sets(img_paths, path_arg_to_index_dict):
    x_set = []
    y_set = []

    for path in img_paths:
        keyword_found = False
        for keyword, label_index in path_arg_to_index_dict.items():

            if keyword in path:

                if not keyword_found:
                    x_set.append(path)
                    y_set.append(label_index)
                    keyword_found = True
                else:
                    raise ValueError('Data path {} contains two or more keywords'.format(path))

        if not keyword_found:
            raise ValueError('Data path {} does not contain a given keyword'.format(path))

    return x_set, y_set

def generate_data_loader(chunk_file_path, chunks_to_use, path_arg_to_index_dict, batch_size, shuffle, num_workers, enable_color_jitter=False):

    img_paths = load_chunk_file(chunk_file_path, chunks_to_use)
    x_set, y_set = generate_x_y_sets(img_paths, path_arg_to_index_dict)

    patch_dataset = PatchDataset(x_set, y_set, enable_color_jitter=enable_color_jitter)
    dataloader = DataLoader(patch_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader

def gpu_selector(gpu_to_use = -1):

    gpu_to_use = -1 if gpu_to_use == None else gpu_to_use

    deviceCount = nvmlDeviceGetCount()

    if gpu_to_use < 0:
        print("Auto selecting GPU") 
        gpu_free_mem = 0

        for i in range(deviceCount):
            handle = nvmlDeviceGetHandleByIndex(i)
            mem_usage = nvmlDeviceGetMemoryInfo(handle)
            if gpu_free_mem < mem_usage.free:
                gpu_to_use = i
                gpu_free_mem = mem_usage.free
            print("GPU: {} \t Free Memory: {}".format(i, mem_usage.free))

    print("Using GPU {}".format(gpu_to_use))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_to_use)
    return gpu_to_use

def setup_log_file(log_folder_path, log_name):
    os.makedirs(log_folder_path, exist_ok = True)
    l_path = os.path.join(log_folder_path, "log_{}.txt".format(log_name))
    sys.stdout = aim_logger.Logger(l_path)
    return

def config_checker(config, look_for):
    if look_for in config:
        return config[look_for] 
    else:
        raise ValueError("{} not in config".format(look_for))

def load_config_file(config_file_path):

    with open(config_file_path) as json_file:
        data = json.load(json_file)

    return data

def load_log_file(log_file_path):

    config_regex = "({[\d\S\W]*})*"
    instance_name_regex = "Instance name: ([\w\S]*)"
    best_iter_regex = "Peak Acc % with [\d.]* at (\d*)"

    with open(log_file_path) as log_file:
        data = log_file.read()

    config_search = re.search( config_regex, data)
    config = config_search.group(1)
    
    name_search = re.search( instance_name_regex, data)
    name = name_search.group(1)

    # get best acc % iter
    best_itr_search = re.search( best_iter_regex, data)
    best_itr = best_itr_search.group(1)

    o = {"config" : json.loads(config), "instance_name" : name, "peak_itr" : best_itr}

    return o

def get_instance_name():
    return 

def parse_input():

    parser = argparse.ArgumentParser()

    parser.add_argument("--log_file_location", help = "local location of log file", required = True)
    parser.add_argument("--config_file_location", help = "local location of config file. Only used for creating docker binds. Do not use if not using docker.", required = False)
    args = parser.parse_args()

    inputs = {}

    inputs['log_file_location'] = args.log_file_location

    return inputs

def main():

    cc = config_checker

    inputs = parse_input()

    log_file = load_log_file(inputs['log_file_location'])

    config = log_file['config']

    test_instance_name = log_file["instance_name"]

    if "test_log_folder_location" in config:
        setup_log_file(config["test_log_folder_location"], test_instance_name)
    
    print(json.dumps(config, indent=4))
    print("Instance name: {}".format(test_instance_name))

    gpu_to_use = gpu_selector(cc(config, "gpu_to_use"))

    test_set = generate_data_loader(cc(config, "chunk_file_location"), cc(config, "test_chunks"), cc(config, "path_value_to_index"), cc(config, "batch_size"), cc(config, "test_shuffle"), cc(config, "patch_workers"), enable_color_jitter=False)

    model = build_model(cc(config, "model_config"))
    model.load_state(cc(config, "model_save_location"), test_instance_name, log_file["peak_itr"])
    # load model

    results = test(model, test_set, cc(config, "path_value_to_index"))

    config["test_results"] = results

    # save results
    config_w_results_json = json.dumps(config, indent=4)
    
    os.makedirs(config["test_results_location"], exist_ok=True)

    with open(os.path.join(config["test_results_location"], test_instance_name + "_test_results.json"), "w") as results_json:
        results_json.write(config_w_results_json)

    return

if __name__ == "__main__":
    main()

