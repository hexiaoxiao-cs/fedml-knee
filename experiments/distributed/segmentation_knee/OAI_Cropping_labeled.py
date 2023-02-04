import SimpleITK as sitk
import os
import numpy as np
import tqdm


# list_bl="../../../data/OAI/bl_sag_3d_dess_with_did.txt"
# list_12m="../../../data/OAI/12m_sag_3d_dess_with_did.txt"

list_label="../../../data/OAI/label_file_did.txt"
root_dir="../../../data/OAI/"
label={0:[],1:[],2:[],3:[],4:[],5:[]}
def bbox2_3d(img):
    r=np.any(img,axis=(1,2))
    c=np.any(img,axis=(0,2))
    z=np.any(img,axis=(0,1))
    rmin,rmax=np.where(r)[0][[0,-1]]
    cmin,cmax=np.where(c)[0][[0,-1]]
    zmin,zmax=np.where(z)[0][[0,-1]]
    return rmin,rmax,cmin,cmax,zmin,zmax

with open(list_label,"r") as f:
    for line in f.readlines():
        path_i=line.split()
        # print(path_i)
        name=os.path.basename(path_i[0])
        names=name.split("_")
        # to_exclude_pid.append(names[0])
        # to_exclude_serial.append(names[1])
        subject_name=names[0]+"_"+names[1]
        p_l=os.path.join(root_dir,path_i[0])
        if "bl" in p_l:
            p_i = os.path.join(root_dir, "img_with_mask", "bl_sag_3d_dess_mhd", subject_name + ".mha")
        else:
            p_i = os.path.join(root_dir, "img_with_mask", "12m_sag_3d_dess_mhd", subject_name + ".mhd")
        #print(path_i[1])
        label[int(path_i[1])].append((subject_name,p_i,p_l))
processing=0

filter=sitk.CropImageFilter()
#field of view 144,224,272
roi=np.asarray([144,224,272])
size_whole=np.asarray([160,384,384],dtype=np.uint)
width=roi/2
resampler = sitk.FlipImageFilter()
flipAxes = [False, False, True]
resampler.SetFlipAxes(flipAxes)
# print(width)
# ct=0
for (name,path,p_label) in tqdm.tqdm(label[processing]):
    # if "11292809" not in name:
    #     continue
    img=sitk.ReadImage(path)
    lbl=sitk.ReadImage(p_label)
    if "RIGHT" in path:
        img=resampler.Execute(img)
        lbl=resampler.Execute(lbl)
    img_np=sitk.GetArrayFromImage(img)
    lbl_np=sitk.GetArrayFromImage(lbl)

    # print(img.GetSize())
    # print(img_np.shape)
    # print(bbox[name])
    #Get Center:
    # try:
    #     lower=np.asarray(bbox[name][3:6],dtype=float)
    #     upper=np.asarray(bbox[name][6:],dtype=float)
    # except:
    #     continue
    a,b,c,d,e,f=bbox2_3d(lbl_np)
    lower=np.asarray((a,c,e),dtype=float)
    # center=(upper+lower)/2
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
    cropped_lbl=filter.Execute(lbl)
    # print(cropped.GetSize())
    # print(cropped_lbl.GetSize())
    # print(sitk.GetArrayFromImage(cropped).shape)
    sitk.WriteImage(cropped,os.path.join(root_dir,"cropped_new_new_labeled",name+"_img.mha"))
    sitk.WriteImage(cropped_lbl,os.path.join(root_dir,"cropped_new_new_labeled",name+"_lbl.mha"))

# print(ct/len(unlabel[processing]))