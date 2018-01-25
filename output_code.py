import os

valid_suffix = ['py', 'md', 'css', 'html']

fw = open('allcode.txt', 'w')

def output(fpath):
    if fpath.split('.')[-1] in valid_suffix:
        fw.write("\n---------------------------------------------\n")
        fw.write(fpath)
        print(fpath)
        fw.write("\n---------------------------------------------\n")
        for line in open(fpath):
            fw.write(line)
        fw.write("\n")

def dfs(root_dir):
    for fname in os.listdir(root_dir):
        fpath = os.path.join(root_dir, fname)
        if os.path.isfile(fpath):
            output(fpath)
        else:
            assert os.path.isdir(fpath)
            dfs(fpath)

dfs('./')
fw.close()
