# coding: utf-8
import glob
import csv

def fname_to_csv(source):
    namelist = glob.glob(source + "*_original.png")
    path_length = len(source)
    with open(source + "r2_rms.csv", "w") as f:
        writer = csv.writer(f, lineterminator='\n')
        for name in namelist:
            split = name[path_length:].replace("r2", "").replace("rms_", "").replace("_original.png", "").split("_")
            writer.writerow(split)


if __name__ == '__main__':
    source = "G:\\Toyota\\Data\\Incompressible_Invicid\\fig_post\\"
    fname_to_csv(source)
