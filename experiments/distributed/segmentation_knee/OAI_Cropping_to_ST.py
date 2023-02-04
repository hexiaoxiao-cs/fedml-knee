import SimpleITK as sitk
import os
import numpy as np
import tqdm


list_bl="../../../data/OAI/bl_sag_3d_dess_with_did.txt"
list_12m="../../../data/OAI/12m_sag_3d_dess_with_did.txt"
root_dir="../../../data/OAI/"
unlabel={0:[],1:[],2:[],3:[],4:[],5:[]}
processing=4
with open(list_bl,"r") as f:
    for line in f.readlines():
        terms=line.split()
        # if (terms[0] in to_exclude_pid) & (terms[1] in to_exclude_serial):
        #     continue
        # else:
        unlabel[int(terms[3])].append((terms[0]+"_"+terms[1],os.path.join(root_dir, "bl", terms[0] + "_" + terms[1] + "_" + terms[2] + ".mha")))
with open(list_12m,"r") as f:
    for line in f.readlines():
        terms=line.split()
        # if (terms[0] in to_exclude_pid) & (terms[1] in to_exclude_serial):
        #     continue
        # else:
        unlabel[int(terms[3])].append((terms[0]+"_"+terms[1],os.path.join(root_dir, "12m", terms[0] + "_" + terms[1] + "_" + terms[2] + ".mha")))

bbox=dict()

with open("bbox_"+str(processing)+".csv","r") as f:
    for line in f.readlines():
        if "failed" in line:
            continue
        terms=line.split(",")
        if terms[2]=="0":
            bbox[terms[1]]=terms
            # print(terms[1])

filter=sitk.CropImageFilter()
#field of view 144,224,272
roi=np.asarray([32,464,464])
size_whole=np.asarray([160,384,384],dtype=np.uint)
width=roi/2
resampler = sitk.FlipImageFilter()
flipAxes = [False, False, True]
resampler.SetFlipAxes(flipAxes)
# print(width)
# ct=0
for (name,path) in tqdm.tqdm(unlabel[processing]):
    # if "11292809" not in name:
    #     continue
    img=sitk.ReadImage(path)
    img_np=sitk.GetArrayFromImage(img)
    if "RIGHT" in path:
        img=resampler.Execute(img)
    # print(img.GetSize())
    # print(img_np.shape)
    # print(bbox[name])
    #Get Center:
    try:
        lower=np.asarray(bbox[name][3:6],dtype=float)
        upper=np.asarray(bbox[name][6:],dtype=float)
    except:
        continue
    center=(upper+lower)/2
    # print("Upper_orig",upper)
    # print("Lower_orig",lower)
    # print("Center",center)
    b_u=lower
    b_d=lower+roi
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
    # print("Post",b_u)
    # print("Post",b_d)
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
    sitk.WriteImage(cropped,os.path.join(root_dir,"cropped_new_new",name+".mha"))

# print(ct/len(unlabel[processing]))