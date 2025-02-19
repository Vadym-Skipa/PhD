import os
import re


def looking_for_all_inner_files_in_the_directory(directory_path):
    outcome_list = []
    instances = os.listdir(directory_path)
    for instance in instances:
        if os.path.isfile(os.path.join(directory_path, instance)):
            outcome_list.append(instance)
        elif os.path.isdir(os.path.join(directory_path, instance)):
            dir_outcome_list = looking_for_all_inner_files_in_the_directory(os.path.join(directory_path, instance))
            temp_outcome_list = [os.path.join(instance, file) for file in dir_outcome_list]
            outcome_list.extend(temp_outcome_list)
    return outcome_list


def looking_for_specific_inner_files_in_the_directory(directory_path, regex_pattern):
    outcome_list = []
    instances = os.listdir(directory_path)
    for instance in instances:
        if os.path.isfile(os.path.join(directory_path, instance)):
            if re.search(regex_pattern, instance) is not None:
                outcome_list.append(instance)
        elif os.path.isdir(os.path.join(directory_path, instance)):
            dir_outcome_list = looking_for_specific_inner_files_in_the_directory(os.path.join(directory_path, instance),
                                                                                 regex_pattern)
            temp_outcome_list = [os.path.join(instance, file) for file in dir_outcome_list]
            outcome_list.extend(temp_outcome_list)
    return outcome_list