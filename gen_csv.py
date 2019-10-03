import os
import sys

def get_file_path(path):
    ret = []
    for p, d, fs in os.walk(path):
        for f in fs:
            ret.append(os.path.join(p, f))
    return ret

def run(file_paths, SAVE_PATH):
    out = open(SAVE_PATH, "w")

    for i in file_paths:
        for j in get_file_path(i[0]):
            out.write(f"{j},{i[1]}\n")

    out.close()

if __name__ == "__main__":
    pass