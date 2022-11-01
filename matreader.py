import scipy.io
from pprint import pprint
mat = scipy.io.loadmat("aplant0_aplant1.mat")

pprint(mat)

# print(mat['ic1'])
print(mat['ic1'][0][0][-1].shape)
print(mat['ic2'][0][0][-1].shape)
print(mat['table'].shape)
print(mat['im1'][0][0][-1].shape)
print(mat['im2'][0][0][-1].shape)
# print(mat['ic2'][-1].shape)
# print(mat['im1'].shape)
# print(mat['im2'].shape)
