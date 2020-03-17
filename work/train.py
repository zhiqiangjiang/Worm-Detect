# -*- coding: utf-8 -*-

import time
import os
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable

from reader import data_loader, test_data_loader, multithread_loader
from yolov3 import YOLOv3

# train.py
# 提升点： 可以改变anchor的大小，注意训练和测试时要使用同样的anchor
#ANCHORS = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
ANCHORS = [7, 10, 12, 22, 24, 17, 22, 45, 46, 33, 43, 88, 85, 66, 115, 146, 275, 240]
ANCHOR_MASKS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

IGNORE_THRESH = .7
NUM_CLASSES = 7

TRAINDIR = 'insects/train'
VALIDDIR = 'insects/val'

def get_lr(base_lr = 0.0001, lr_decay = 0.1):
    bd = [10000, 20000]
    lr = [base_lr, base_lr * lr_decay, base_lr * lr_decay * lr_decay]
    learning_rate = fluid.layers.piecewise_decay(boundaries=bd, values=lr)
    return learning_rate
    
# train.py
if __name__ == '__main__':
    with fluid.dygraph.guard():
        model = YOLOv3('yolov3', num_classes = NUM_CLASSES, is_train=True)
        #opt = fluid.optimizer.Momentum(
        #             learning_rate=0.001,  #提升点：可以调整学习率，或者设置学习率衰减
        #             momentum=0.9)   # 提升点： 可以添加正则化项
        #learning_rate = get_lr()
        #opt = fluid.optimizer.Momentum(
        #             learning_rate=learning_rate,
        #             momentum=0.9，
        #             regularization=fluid.regularizer.L2Decay(0.0005))  #创建优化器
        opt = fluid.optimizer.AdamOptimizer(learning_rate=0.001,regularization=fluid.regularizer.L2Decay(0.0005))
             
        train_loader = multithread_loader(TRAINDIR, batch_size= 10, mode='train')
        valid_loader = multithread_loader(VALIDDIR, batch_size= 10, mode='valid')

        MAX_EPOCH = 300  # 提升点： 可以改变训练的轮数
        for epoch in range(MAX_EPOCH):
            for i, data in enumerate(train_loader()):
                img, gt_boxes, gt_labels, img_scale = data
                gt_scores = np.ones(gt_labels.shape).astype('float32')
                gt_scores = to_variable(gt_scores)
                img = to_variable(img)
                gt_boxes = to_variable(gt_boxes)
                gt_labels = to_variable(gt_labels)
                outputs = model(img)
                loss = model.get_loss(outputs, gt_boxes, gt_labels, gtscore=gt_scores,
                                      anchors = ANCHORS,
                                      anchor_masks = ANCHOR_MASKS,
                                      ignore_thresh=IGNORE_THRESH,
                                      use_label_smooth=False)

                loss.backward()
                opt.minimize(loss)
                model.clear_gradients()
                if i % 40 == 0:
                    timestring = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
                    print('{}[TRAIN]epoch {}, iter {}, output loss: {}'.format(timestring, epoch, i, loss.numpy()))

            # save params of model
            if (epoch % 4 == 0) or (epoch == MAX_EPOCH -1):
                fluid.save_dygraph(model.state_dict(), 'yolo_epoch{}'.format(epoch))
                 # 每10个epoch结束之后在验证集上进行测试
                model.eval()
                for i, data in enumerate(valid_loader()):
                    img, gt_boxes, gt_labels, img_scale = data
                    gt_scores = np.ones(gt_labels.shape).astype('float32')
                    gt_scores = to_variable(gt_scores)
                    img = to_variable(img)
                    gt_boxes = to_variable(gt_boxes)
                    gt_labels = to_variable(gt_labels)
                    outputs = model(img)
                    loss = model.get_loss(outputs, gt_boxes, gt_labels, gtscore=gt_scores,
                                      anchors = ANCHORS,
                                      anchor_masks = ANCHOR_MASKS,
                                      ignore_thresh=IGNORE_THRESH,
                                      use_label_smooth=False)
                    if i % 3 == 0:
                        timestring = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
                        print('{}[VALID]epoch {}, iter {}, output loss: {}'.format(timestring, epoch, i, loss.numpy()))
                model.train()


