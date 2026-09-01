# import scipy.io as sio

# mat = sio.loadmat('data/training_dataset/training_data/annotations/Buffy_1.mat')

# # print(mat.keys())
# # print(mat['boxes'])
# # print(mat['boxes'].shape)
# # print(mat['boxes'][0, 0])


# box = mat['boxes'][0,0]

# struct = box[0,0]

# print(struct)

import scipy.io as sio
mat = sio.loadmat('../data/training_dataset/training_data/annotations/Buffy_1.mat',
                  squeeze_me=True, struct_as_record=False)
boxes = mat['boxes']
print(f'Number of hands: {len(boxes) if hasattr(boxes, "__len__") else 1}')
for i, box in enumerate(boxes if hasattr(boxes, "__len__") else [boxes]):
    print(f'  Hand {i}: a={box.a}, b={box.b}, c={box.c}, d={box.d}')