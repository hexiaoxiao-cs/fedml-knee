import glob
import shutil
import os

for i in glob.glob("./ST_prediction/*_img.mha"):
    #org seg
    name=os.path.basename(i)
    shutil.copy(i,i.replace("ST_prediction","ST_format").replace("img","org"))
    shutil.copy(i.replace("img","pred"),i.replace("img","seg").replace("ST_prediction","ST_format"))
