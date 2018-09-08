import numpy as np
import sys
import caffe
import copy

net = caffe.Net('yolo_conv21.prototxt', caffe.TEST)
print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))
params = ['conv21_1', 'conv21_2', 'bn21_1', 'bn21_2', 'scale21_1', 'scale21_2']

conv_params = {pr: (net.params[pr][0].data) for pr in params}


for fc in params:
    print('{} weights are {} dimensional and biases are 0 dimensional'.format(fc, conv_params[fc].shape))


net_ref = caffe.Net('yolo_conv21_ref.prototxt', 'yolo_conv21_ref.caffemodel', caffe.TEST)
print("for the reference model: blobs {}\nparams {}".format(net_ref.blobs.keys(), net_ref.params.keys()))
params_ref = ['conv21', 'bn21', 'scale21']
conv_params_ref = {pr: (net_ref.params[pr][0].data)  for pr in params_ref}

for fc in params_ref:
    print('for reference model: {} weights are {} dimensional and biases are 0 dimensional'.format(fc, conv_params_ref[fc].shape))

for param in net.params.keys():
    
    if param not in params:
        for i in range(len(net.params[param])):
            net.params[param][i].data[:] = copy.deepcopy(net_ref.params[param][i].data[:])
    else:
        print(param)
        layer = param[:-2]
        num = int(param[-1:])
        if layer == 'conv21':
            for i in range(len(net.params[param])):
                part = int(net_ref.params[layer][i].data.shape[1] / 2)
                # print(layer+' '+str(part))
                
                net.params[param][i].data[:] = copy.deepcopy(net_ref.params[layer][i].data[:,part*(num-1):part*num, :, :])
                print(param+' shape is'+str(net.params[param][i].data.shape))
                # print(net.params[param][0].data[:, part*(num-1):part*num, :, :])
                
        else:
            for i in range(len(net.params[param])):
                # part = int(net_ref.params[layer][i].data.shape[0]/2)
                # only change the convolution params, others remains the same since it's the same dimension with the output
                # net.params[param][i].data[:] = net_ref.params[layer][i].data[part*(num-1):part*num]
                if (layer == 'bn21' and i==0) or (layer == 'scale21' and i==1):
                    print(param+' before is:')
                    print(net_ref.params[layer][i].data[:])
                    net.params[param][i].data[:] = copy.deepcopy(0.5*net_ref.params[layer][i].data[:])
                    print(param+' now is:')
                    print(net.params[param][i].data[:])
                    continue
                # if layer == 'scale21' and i==1:
                #     print(param+' before is:')
                #     print(net_ref.params[layer][i].data[:])
                #     net.params[param][i].data[:] = copy.deepcopy(0.5*net_ref.params[layer][i].data[:])
                #     print(param+' now is:')
                #     print(net.params[param][i].data[:])
                #     continue
                net.params[param][i].data[:] = copy.deepcopy(net_ref.params[layer][i].data[:])
                print(param+' shape is'+str(net.params[param][i].data.shape))
                print(net.params[param][i].data[:])

print('transplant done...')
# print(net.params['conv21_2'][0].data[:])

# data = net.params['conv21_1'][0].data[:]
# with open('conv21_1.txt', 'w') as outfile:
#     outfile.write('# blob shape: {}\n'.format(data.shape))
#     for kernel in data:
#         outfile.write('# New kernel\n')
#         outfile.write('kernel shape: {}\n'.format(kernel.shape))
#         for index, feat_layer in enumerate(kernel):
#             outfile.write('# New feat_layer of {}\n'.format(index))
#             outfile.write('feat_layer shape: {}\n'.format(feat_layer.shape))
#             np.savetxt(outfile, feat_layer)
            
# data = net_ref.params['conv21'][0].data[:]
# with open('conv21.txt', 'w') as outfile:
#     outfile.write('# blob shape: {}\n'.format(data.shape))
#     for kernel in data:
#         outfile.write('# New kernel\n')
#         outfile.write('kernel shape: {}\n'.format(kernel.shape))
#         for index, feat_layer in enumerate(kernel):
#             outfile.write('# New feat_layer of {}\n'.format(index))
#             outfile.write('feat_layer shape: {}\n'.format(feat_layer.shape))
#             np.savetxt(outfile, feat_layer)
        
# data = net.params['conv21_2'][0].data[:]
# with open('conv21_2.txt', 'w') as outfile:
#     outfile.write('# blob shape: {}\n'.format(data.shape))
#     for kernel in data:
#         outfile.write('# New kernel\n')
#         outfile.write('kernel shape: {}\n'.format(kernel.shape))
#         for index, feat_layer in enumerate(kernel):
#             outfile.write('# New feat_layer of {}\n'.format(index))
#             outfile.write('feat_layer shape: {}\n'.format(feat_layer.shape))
#             np.savetxt(outfile, feat_layer)

net.save('group_conv.caffemodel')






