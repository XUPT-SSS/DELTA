import csv
import json
import os
import re
import time
import sys


def preprocess():
    work_dir = os.path.abspath('.')
    root_dir = os.path.join(work_dir, 'json_shop')
    for pefile_dir in os.listdir(root_dir):
        temp = pefile_dir
        pefile_dir = os.path.join(root_dir , pefile_dir)
        for file in os.listdir(pefile_dir):
            dict = {}
            with open(pefile_dir + '/' + file) as f:
                # f_csv = csv.reader(f, delimiter='\n', quoting=csv.QUOTE_NONE)
                f_json = json.load(f)
                for i in range(len(f_json)):
                    row = f_json[i]
                    if i != 0:
                        cmp = f_json[i-1]
                        cmp_label = cmp[0]
                    label = row[0]
                    if 'good' in label and 'bad' in cmp_label:
                        temp_label = cmp_label
                        temp_label = temp_label.replace("bad", "good")
                        label = temp_label
                    if ('bad' in label) or ('good' in label):
                        # 定义正则表达式模式
                        pattern = r'CWE\d+_.+?_(\d+)'
                        # 使用正则表达式查找匹配的部分
                        match = re.search(pattern, label)
                        # print(match)
                        # 如果找到匹配项，则打印匹配到的内容
                        if match:
                            result = match.group(0)
                            # print("匹配的部分:", result)
                            if 'Sink' in label:
                                if 'bad' in label:
                                    label = result + "_badSink"
                                if 'good' in label:
                                    label = result + "_goodSink"
                            elif 'bad' in label:
                                label = result + "_bad"
                            elif 'good' in label:
                                label = result + "_good"
                        # else:
                        #     print("未找到匹配项")
                        instru_list = re.split('\n', row[1])
                        f_res = []
                        if len(instru_list) <= 30:
                            continue
                        for sub_instr in instru_list:
                            # 空格替换
                            tmp = re.sub(' +', "-", sub_instr, 1)
                            tmp = re.sub(' ', "", tmp)
                            # 删除分号后内容
                            tmp = re.sub(';.*', "", tmp)
                            # 存储地址使用MEM代替
                            tmp = re.sub('=.*', "MEM", tmp)
                            tmp = re.sub('#.*\+.*[^\]]', "MEM", tmp)
                            # 替换offsetlabel
                            tmp = re.sub('offset.+', "offset", tmp)
                            tmp = re.sub('byte_[0-9a-fA-F]+', 'BYTE', tmp)
                            tmp = re.sub('qword_[0-9a-fA-F]+', 'QWORD', tmp)
                            tmp = re.sub('[a-z]s:[^,]*', "MEM", tmp)
                            tmp = re.sub('\+.*]', "+MEM]", tmp)
                            ok = re.match('lea-', tmp)
                            if ok:
                                tmp_list = re.split(',', tmp)
                                tmp_sublist = re.split('-', tmp_list[0])
                                if len(tmp_sublist[1]) > 3:
                                    tmp = 'lea-MEM,' + tmp_list[1]
                                elif len(tmp_list[1]) > 3:
                                    tmp = tmp_list[0] + ',MEM'
                            # 基本块使用LOC代替
                            tmp = re.sub('loc.*', "LOC", tmp)

                            tmp = re.sub('shortLOC', "LOC", tmp)
                            # 函数使用FUN代替
                            # tmp = re.sub('BL-.*', "BL-FUN", tmp)
                            tmp = re.sub('jmp-.*', "jmp-LOC", tmp)
                            tmp = re.sub('call-.*', "call-FUN", tmp)
                            # 立即数使用IMM代替
                            t = re.search('#?0[xX][0-9a-fA-F]+', tmp)
                            if t:
                                s = t.group(0)
                                if s[0] == '#':
                                    s = s[3:]
                                else:
                                    s = s[2:]
                                num = 0
                                for i in range(len(s)):
                                    num = num * 16 + int(s[i], 16)
                                if num > 1000:
                                    tmp = re.sub('#?0[xX][0-9a-fA-F]+', "IMM", tmp)
                            tmp = re.sub(',', "-", tmp)
                            f_res.append(tmp)
                    else:
                        continue
                    dict[label] = f_res
                target_dir = os.path.join(work_dir , 'preprocess_shop',temp)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                with open(target_dir + '/'+ file, "w") as cur_file:
                    print("###FileName: " + file)
                    json.dump(dict, cur_file)
    return


def standardize():
    work_dir = os.path.abspath('.')
    pefile_dir = os.path.join(work_dir, 'preprocess_shop')

    key_dict = {}
    file_dict = {}
    for file in os.listdir(pefile_dir):
        with open(pefile_dir + '/' + file) as f:
            func_dict = json.load(f)
            file_name_tmp = file[:-5].split('-')
            file_name = file_name_tmp[-2] + '-' + file_name_tmp[-1] + '-' + file_name_tmp[-3]
            tmp_list = []
            tmp_dict = {}
            for key, value in func_dict.items():
                tmp_list.append(key)
                if tmp_dict.get(key):
                    tmp_dict[key].append(value)
                else:
                    tmp_dict[key] = value

            key_dict[file_name] = tmp_list
            file_dict[file_name] = tmp_dict

    inte_list = list(key_dict.values())[0]
    for item_list in key_dict.values():
        inte_list = list(set(item_list).intersection(set(inte_list)))
    # inte_set = set(inte_list)

    for file_name, func in file_dict.items():
        res_dict = {}
        res_list = [0 for _ in range(len(inte_list))]
        for i in range(len(inte_list)):
            func_name = inte_list[i]
            # print(type(func[func_name]))
            res_dict[func_name] = func[func_name]
            # print(func[func_name])

        with open("standard_shop/" + file_name + ".json", "w") as cur_file:
            print("###FileName: " + file_name)
            json.dump(res_dict, cur_file)
    return


def main():
    preprocess()

if __name__ == '__main__':
    main()

