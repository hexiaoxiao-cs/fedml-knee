from concurrent.futures import process
import SimpleITK as sitk
import os
import numpy as np
import tqdm
import sys
from multiprocessing import Pool
import glob
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../FedML")))
from data_preprocessing.sensetime.utils import *
# list_bl="../../../data/OAI/bl_sag_3d_dess_with_did.txt"
# list_12m="../../../data/OAI/12m_sag_3d_dess_with_did.txt"
root_dir="../../../data/sensetime/"
# unlabel={0:[],1:[],2:[],3:[],4:[],5:[]}
# processing=4
# with open(list_bl,"r") as f:
#     for line in f.readlines():
#         terms=line.split()
#         # if (terms[0] in to_exclude_pid) & (terms[1] in to_exclude_serial):
#         #     continue
#         # else:
#         unlabel[int(terms[3])].append((terms[0]+"_"+terms[1],os.path.join(root_dir, "bl", terms[0] + "_" + terms[1] + "_" + terms[2] + ".mha")))
# with open(list_12m,"r") as f:
#     for line in f.readlines():
#         terms=line.split()
#         # if (terms[0] in to_exclude_pid) & (terms[1] in to_exclude_serial):
#         #     continue
#         # else:
#         unlabel[int(terms[3])].append((terms[0]+"_"+terms[1],os.path.join(root_dir, "12m", terms[0] + "_" + terms[1] + "_" + terms[2] + ".mha")))

bbox=dict()

with open("bbox_ST_lbl.csv","r") as f:
    for line in f.readlines():
        if "failed" in line:
            continue
        terms=line.split(",")
        if terms[2]=="0":
            bbox[terms[1]]=terms
            # print(terms[1])

def processing(path):
    name=os.path.basename(path)
    img=sitk.ReadImage(path)
    img_np=sitk.GetArrayFromImage(img)
    img_np = pad_crop(img_np, (144,400,400), mode='constant', value=0)
    img_new=sitk.GetImageFromArray(img_np)
    img_new.SetOrigin(img.GetOrigin())
    img_new.SetDirection(img.GetDirection())
    img_new.SetSpacing(img.GetSpacing())
    img=img_new
    # if "RIGHT" in path:
    #     img=resampler.Execute(img)
    # print(img.GetSize())
    # print(img_np.shape)
    # print(bbox[name])
    #Get Center:
    try:
        lower=np.asarray(bbox[name][3:6],dtype=float)
        upper=np.asarray(bbox[name][6:],dtype=float)
    except:
        return
    center=(upper+lower)/2
    # print("Upper_orig",upper)
    # print("Lower_orig",lower)
    # print("Center",center)
    # Using lower as base
    # b_u=lower
    # b_d=lower+roi
    # Using center as base
    b_u=center-width
    b_d=center+width
    # size_bbox=upper-lower
    # for i in range(3):
    #     if size_bbox[i]>roi[i]:
    #         print(name)
    #         ct=ct+1
    #         break
    # print("Prior",center-width)
    # print("Prior",center+width)
    #Upper bound move down if upper<0
    for i in range(3):
        if b_u[i]<0:
            b_d[i]-=b_u[i]
            b_u[i]=0
    #Lower bound move up if lower>wholesize
    for i in range(3):
        if b_d[i]>size_whole[i]:
            b_u[i]=b_u[i]-b_d[i]+size_whole[i]
            b_d[i]=size_whole[i]
    # print("Post_u",b_u)
    # print("Post_d",b_d)
    b_u=b_u.astype(int)
    b_d=b_d.astype(int)
    # print("Upper",b_u)
    # print("Lower",b_d)
    # print(size_whole-b_d)
    to_lower=np.flip(size_whole-b_d)
    to_upper=np.flip(b_u)
    # print(to_lower)
    # print(to_upper)
    filter.SetLowerBoundaryCropSize([int(i) for i in to_upper])
    filter.SetUpperBoundaryCropSize([int(i) for i in to_lower])
    cropped=filter.Execute(img)
    # print(cropped.GetSize())
    # print(sitk.GetArrayFromImage(cropped).shape)
    sitk.WriteImage(cropped,os.path.join(root_dir,"cropped_img_new_unlbl",name+".mha"))

if __name__=="__main__":
    filter=sitk.CropImageFilter()
    #field of view 144,224,272
    roi=np.asarray([144,224,272])
    size_whole=np.asarray([144,400,400],dtype=np.uint)
    width=roi/2
    resampler = sitk.FlipImageFilter()
    flipAxes = [False, False, True]
    resampler.SetFlipAxes(flipAxes)
    # print(width)
    # ct=0
    pool=Pool(70)
    pool.map(processing,glob.glob(os.path.join(root_dir,"ST_full_reso_nolbl","*.nii.gz")))
    # for path in tqdm.tqdm(glob.glob(os.path.join(root_dir,"ST_full_reso_nolbl","*.nii.gz"))):
    #     # if "11292809" not in name:
    #     #     continue
    #     name=os.path.basename(path)
    #     img=sitk.ReadImage(path)
    #     img_np=sitk.GetArrayFromImage(img)
    #     img_np = pad_crop(img_np, (144,400,400), mode='constant', value=0)
    #     img_new=sitk.GetImageFromArray(img_np)
    #     img_new.SetOrigin(img.GetOrigin())
    #     img_new.SetDirection(img.GetDirection())
    #     img_new.SetSpacing(img.GetSpacing())
    #     img=img_new
    #     # if "RIGHT" in path:
    #     #     img=resampler.Execute(img)
    #     # print(img.GetSize())
    #     # print(img_np.shape)
    #     # print(bbox[name])
    #     #Get Center:
    #     try:
    #         lower=np.asarray(bbox[name][3:6],dtype=float)
    #         upper=np.asarray(bbox[name][6:],dtype=float)
    #     except:
    #         continue
    #     center=(upper+lower)/2
    #     # print("Upper_orig",upper)
    #     # print("Lower_orig",lower)
    #     # print("Center",center)
    #     # Using lower as base
    #     # b_u=lower
    #     # b_d=lower+roi
    #     # Using center as base
    #     b_u=center-width
    #     b_d=center+width
    #     # size_bbox=upper-lower
    #     # for i in range(3):
    #     #     if size_bbox[i]>roi[i]:
    #     #         print(name)
    #     #         ct=ct+1
    #     #         break
    #     # print("Prior",center-width)
    #     # print("Prior",center+width)
    #     #Upper bound move down if upper<0
    #     for i in range(3):
    #         if b_u[i]<0:
    #             b_d[i]-=b_u[i]
    #             b_u[i]=0
    #     #Lower bound move up if lower>wholesize
    #     for i in range(3):
    #         if b_d[i]>size_whole[i]:
    #             b_u[i]=b_u[i]-b_d[i]+size_whole[i]
    #             b_d[i]=size_whole[i]
    #     # print("Post_u",b_u)
    #     # print("Post_d",b_d)
    #     b_u=b_u.astype(int)
    #     b_d=b_d.astype(int)
    #     # print("Upper",b_u)
    #     # print("Lower",b_d)
    #     # print(size_whole-b_d)
    #     to_lower=np.flip(size_whole-b_d)
    #     to_upper=np.flip(b_u)
    #     # print(to_lower)
    #     # print(to_upper)
    #     filter.SetLowerBoundaryCropSize([int(i) for i in to_upper])
    #     filter.SetUpperBoundaryCropSize([int(i) for i in to_lower])
    #     cropped=filter.Execute(img)
    #     # print(cropped.GetSize())
    #     # print(sitk.GetArrayFromImage(cropped).shape)
    #     sitk.WriteImage(cropped,os.path.join(root_dir,"cropped_img_unlbl",name+".mha"))

    # # print(ct/len(unlabel[processing]))