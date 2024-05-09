
import json
def trans_formart(file_path):

    label_index={}
    label_index['<start>'] = 0
    label_index['<eos>'] = 1
    label_index['<pad>'] = 2

    seg_label_index={}
    seg_label_index['<start>'] = 0
    seg_label_index['<eos>'] = 1
    seg_label_index['<pad>'] = 2
    seg_label_index["O"]=3
    seg_label_index["B"] = 4
    seg_label_index["I"] = 5
    index=3
    max_len = 0

    with open(file_path) as file:
        for line in file:
            wordlist,labellist,seglist=line.strip().split('|||')
            max_len=max(max_len,len(wordlist.split()))
            for label in labellist.split():
                if label not in label_index:
                    label_index[label]=index
                    index+=1

    print('max_len={}'.format(max_len))
    with open(file_path.replace(".txt","label_dict.json"),'w') as file:
        json.dump(label_index,file,indent=4,ensure_ascii=False)
    with open(file_path.replace(".txt","seg_dict.json"),'w') as file:
        json.dump(seg_label_index,file,indent=4,ensure_ascii=False)

if __name__ == '__main__':
    # trans_formart('./people_daily/train_cutword.txt')
    trans_formart('./data/MSRA/testright1_format.txt')