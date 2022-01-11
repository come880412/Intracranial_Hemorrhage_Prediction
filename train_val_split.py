import numpy as np
import os
import tqdm
import random

label_name_dict = {0:'intraventricular', 1:'subarachnoid', 2:'subdural', 3:'intraparenchymal', 4:'epidural'} 

def train_val_split(csv_path, train_val_ratio, nosick_ratio, train_save_name, val_save_name, random_split):
    train_label_count = {'0':0, '1':0, '2':0, '3':0, '4':0}
    val_label_count = {'0':0, '1':0, '2':0, '3':0, '4':0}
    train_csv_file = np.loadtxt(csv_path, delimiter=',', dtype=np.str)[1:]
    study_sick_instance_uid = []
    study_nosick_instance_uid = []
    sick_file_name = []
    no_sick_file_name = []
    for meta_data in tqdm.tqdm(train_csv_file):
        SI_uid = meta_data[2]
        file_name = meta_data[1]
        label = meta_data[3:].astype(int)
        sick_determine = np.any(label)
        if sick_determine:
            study_sick_instance_uid.append(SI_uid)
            sick_file_name.append(file_name)
        else:
            no_sick_file_name.append(file_name)
            study_nosick_instance_uid.append(SI_uid)
    if not random_split:
        # no_sick file
        study_nosick_instance_uid = np.array(study_nosick_instance_uid)
        study_nosick_instance_uid = np.unique(study_nosick_instance_uid)
        number_of_nosick_study_uid = len(study_nosick_instance_uid)
        random_num_list = np.random.choice(number_of_nosick_study_uid, int(nosick_ratio * number_of_nosick_study_uid), replace=False)

        train_index = np.array(random_num_list[:int(len(random_num_list) * train_val_ratio)], dtype=int)
        val_index = np.array(random_num_list[int(len(random_num_list) * train_val_ratio):], dtype=int)

        train_nosick_uid = study_nosick_instance_uid[train_index]
        val_nosick_uid = study_nosick_instance_uid[val_index]

        # sick file
        study_sick_instance_uid = np.array(study_sick_instance_uid)
        study_sick_instance_uid = np.unique(study_sick_instance_uid)
        number_of_sick_study_uid = len(study_sick_instance_uid)
        random_num_list = np.random.choice(number_of_sick_study_uid, number_of_sick_study_uid, replace=False)

        train_index = np.array(random_num_list[:int(len(random_num_list) * train_val_ratio)], dtype=int)
        val_index = np.array(random_num_list[int(len(random_num_list) * train_val_ratio):], dtype=int)

        train_sick_uid = study_sick_instance_uid[train_index]
        val_sick_uid = study_sick_instance_uid[val_index]

        train_uid = np.concatenate([train_nosick_uid, train_sick_uid])
        val_uid = np.concatenate([val_nosick_uid, val_sick_uid])

        train_save_csv = []
        val_save_csv = []
        for meta_data in tqdm.tqdm(train_csv_file):
            SI_uid = meta_data[2]
            file_name = meta_data[1]
            label = meta_data[3:]
            if SI_uid in train_uid:
                saved_file = np.insert(label, 0, file_name)
                train_save_csv.append(saved_file)
                for i in range(len(label)):
                    label_sub = int(label[i])
                    if label_sub == 1:
                        train_label_count[str(i)] += 1
            elif SI_uid in val_uid:
                saved_file = np.insert(label, 0, file_name)
                val_save_csv.append(saved_file)
                for i in range(len(label)):
                    label_sub = int(label[i])
                    if label_sub == 1:
                        val_label_count[str(i)] += 1
    else:
        # no_sick file
        no_sick_file_name = np.array(no_sick_file_name)
        number_of_nosick_file_name = len(no_sick_file_name)
        random_num_list = np.random.choice(number_of_nosick_file_name, int(nosick_ratio * number_of_nosick_file_name), replace=False)

        train_index = np.array(random_num_list[:int(len(random_num_list) * train_val_ratio)], dtype=int)
        val_index = np.array(random_num_list[int(len(random_num_list) * train_val_ratio):], dtype=int)

        train_nosick_png = no_sick_file_name[train_index]
        val_nosick_png = no_sick_file_name[val_index]

        # sick file
        sick_file_name = np.array(sick_file_name)
        number_of_sick_file_name = len(sick_file_name)
        random_num_list = np.random.choice(number_of_sick_file_name, number_of_sick_file_name, replace=False)

        train_index = np.array(random_num_list[:int(number_of_sick_file_name * train_val_ratio)], dtype=int)
        val_index = np.array(random_num_list[int(number_of_sick_file_name * train_val_ratio):], dtype=int)

        train_sick_png = sick_file_name[train_index]
        val_sick_png = sick_file_name[val_index]


        train_png = np.concatenate([train_nosick_png, train_sick_png])
        val_png = np.concatenate([val_nosick_png, val_sick_png])

        train_save_csv = []
        val_save_csv = []
        for meta_data in tqdm.tqdm(train_csv_file):
            SI_uid = meta_data[2]
            file_name = meta_data[1]
            label = meta_data[3:]
            if file_name in train_png:
                saved_file = np.insert(label, 0, file_name)
                train_save_csv.append(saved_file)
                for i in range(len(label)):
                    label_sub = int(label[i])
                    if label_sub == 1:
                        train_label_count[str(i)] += 1
            elif file_name in val_png:
                saved_file = np.insert(label, 0, file_name)
                val_save_csv.append(saved_file)
                for i in range(len(label)):
                    label_sub = int(label[i])
                    if label_sub == 1:
                        val_label_count[str(i)] += 1

    np.savetxt(train_save_name, train_save_csv, fmt='%s', delimiter=',')
    np.savetxt(val_save_name, val_save_csv, fmt='%s', delimiter=',')
    print('Statistic!')
    print('------train_count------')
    for key, values in train_label_count.items():
        label_name = label_name_dict[int(key)]
        num_label = values
        print('%s : %d/%d' % ( label_name, num_label, len(train_save_csv)))
    print('------val_count------')
    for key, values in val_label_count.items():
        label_name = label_name_dict[int(key)]
        num_label = values
        print('%s : %d/%d' % ( label_name, num_label, len(val_save_csv)))

if __name__ == '__main__':
    np.random.seed(500)
    train_val_ratio = 0.8
    nosick_ratio = 0.2
    random_split = True
    csv_path = '../dataset/train1_png.csv'
    train_save_name = '../dataset/train.csv'
    val_save_name = '../dataset/val.csv'

    train_val_split(csv_path, train_val_ratio, nosick_ratio, train_save_name, val_save_name, random_split)
    
