import os
import numpy as np

imglist_top_dir = '/media/samsumg_1tb/ApolloScape/ImageLists'
# currently, the training, valid, testing only include road01,02,03 (not 04, maybe we could use 04 for training as well?)

# ['train', 'val', 'test']:
train_list_all = []
label_list_all = []
#for road_idx in [1, 2, 3]:
for road_idx in [1]:
    file_name = 'road%02d_ins_train.lst' % road_idx
    train_list_file = os.path.join(imglist_top_dir, file_name)

    lines = [line.rstrip('\n') for line in open(train_list_file)]
    for line in lines:
        img_name = line.split('\t')[0]
        label_name = line.split('\t')[1]
        train_list_all.append(img_name)
        label_list_all.append(label_name)

# label_name will always be  Camera 5, Camera 6, verify this-- this is not True

time_stamps = [int(x.split('/')[-1][7:16]) for x in train_list_all]
count = 0
for i in range(len(time_stamps)-1):
    if i % 2 == 0:
        ts_5 = time_stamps[i]
        ts_6 = time_stamps[i+1]
        print()
        if ts_5 > ts_6:
            print("Error: " + str(ts_5))
        else:
            count += 1

count = 0
camera_5_time_stamps = np.array([int(x.split('/')[-1][7:16]) for x in train_list_all if x.split('/')[-1][24]=='5'])
camera_5_time_diff = camera_5_time_stamps[1:] - camera_5_time_stamps[:-1]

for i in range(len(camera_5_tim_stamps)-1):
    ts_5 = time_stamps[i]
    ts_6 = time_stamps[i+1]
    print()
    if ts_5 > ts_6:
        print("Error: " + str(ts_5))
    else:
        count += 1

print(count)





