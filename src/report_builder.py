import numpy as np
import os


def clean_txt(file_name, process_type):
    main_path = os.getcwd().replace(os.sep, '/')
    if os.path.exists(main_path + "/out/" + file_name) is not True:
        os.mkdir(main_path + "/out/" + file_name)
    open(f'{main_path}/out/{file_name}/{process_type}.txt', 'w', encoding='utf-8').close() 
        
def write_txt(file_name, process_type, result): 
    main_path = os.getcwd().replace(os.sep, '/')
    with open(f'{main_path}/out/{file_name}/{process_type}.txt', 'a+', encoding='utf-8') as file: 
        for key, value in result.items(): 
            file.write(f'{key}  {value}\n')
    file.close()

def write_matrix(file_name, process_type, matrix):  
    main_path = os.getcwd().replace(os.sep, '/')
    with open(f'{main_path}/out/{file_name}/{process_type}.txt', 'a+', encoding='utf-8') as file: 
        for line in matrix:
            np.savetxt(file, line, fmt='%.3f\t')
