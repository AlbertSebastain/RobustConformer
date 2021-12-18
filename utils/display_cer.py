
from tabulate import tabulate
import os
import argparse
import re

def display_table():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--recog_dir",type=str,help="recog directory")
    parser.add_argument("--test_set_name",action="append",type=str,help="test set name")
    opt = parser.parse_args()
    
    recog_dir = opt.recog_dir
    test_set_names = opt.test_set_name
    test_set_names.sort(key=len,reverse=True)
    model_names=set()
    for root, dirs, file in os.walk(recog_dir):
        break
    for dir in dirs:
        if dir.find("recog_") != -1:
            dir = dir.replace("recog_",'')
        for test_name in test_set_names:
            if dir.find(test_name) != -1:
                model_names.add(dir.replace(test_name+"_",''))
                break
    model_names = list(model_names)
    titles = ["model"]
    test_set_names.reverse()
    for test_name in test_set_names:
        titles.append(test_name)
    Data = [tuple(titles)]
    for model in model_names:
        model_result = [model]
        for test_name in test_set_names:
            model_path = os.path.join(recog_dir,"recog_"+test_name+"_"+model,"result.txt")
            CER = "NOT FOUND"
            if os.path.isfile(model_path):
                with open(model_path,'r',encoding='utf8') as f:
                    lines = f.readlines()
                    lines = "".join(lines)
                match_line = re.search(r"Sum/Avg\s+\|( +[0-9]+(\.[0-9]+)?\s+\|?){8}",lines)
                if match_line is not None:
                    line = match_line.group()
                    CER = re.findall(r"\d+\.?\d+",line)[-2] # Contirbution: Xinlei Zhang provides re
                else:
                    CER = "NOT FOUND"
            model_result.append(CER)
        model_result = tuple(model_result)
        Data.append(model_result)
    string = tabulate(Data,headers='firstrow',tablefmt="grid")
    strings = string.split('\n')
    strings = [str_s.center(100) for str_s in strings]
    string = '\n'.join(strings)
    print("Summary CER [%]".center(100))
    print(string)
if __name__ == '__main__':
    display_table()   
