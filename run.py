import numpy as np
import source.datamanager as dman
from source.network import make_gcn_layer, make_pipgcn
from sklearn import metrics
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(gpus)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)


if __name__ == '__main__':
    dataset = dman.DataSet(dir='./')
    
    def train_generator():
        while True:
            minibatch, terminate = dataset.next_batch(batch_size=128, ttv=0)
            if terminate == True:
                dataset.reset_index()
                minibatch, terminate = dataset.next_batch(batch_size=128, ttv=0)
            node_r, edge_r, hood_r = minibatch['r_vertex'], minibatch['r_edge'], minibatch['r_hood_indices']
            node_l, edge_l, hood_l = minibatch['l_vertex'], minibatch['l_edge'], minibatch['l_hood_indices']
            pair = minibatch['label']
            y = minibatch['label_1hot']
            target = pair[:, 2]
            target[target < 0] = 0
            yield [node_r, edge_r, hood_r, node_l, edge_l, hood_l, pair], target#, []

    def val_generator():
        while True:
            minibatch, terminate = dataset.next_batch(batch_size=128, ttv=2)
            if terminate == True:
                dataset.reset_index()
                minibatch, terminate = dataset.next_batch(batch_size=128, ttv=2)
            node_r, edge_r, hood_r = minibatch['r_vertex'], minibatch['r_edge'], minibatch['r_hood_indices']
            node_l, edge_l, hood_l = minibatch['l_vertex'], minibatch['l_edge'], minibatch['l_hood_indices']
            pair = minibatch['label']
            y = minibatch['label_1hot']
            target = pair[:, 2]
            target[target < 0] = 0
            yield [node_r, edge_r, hood_r, node_l, edge_l, hood_l, pair], target#
    
    # start train
    model = make_pipgcn(70, 8, 1)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=5, restore_best_weights=True, mode='max')
    model.fit(train_generator(), steps_per_epoch=1174, epochs=40, validation_data=val_generator(), validation_steps=289, callbacks=[callback])#, class_weight={0:0.1, 1:1.})
    
    pred_label = []
    label = []

    flag = False
    count = 0
    dataset.reset_index()
    while True:
        minibatch, terminate = dataset.next_batch(batch_size=128, ttv=1)
        if terminate:
            break  
        node_r, edge_r, hood_r = minibatch['r_vertex'], minibatch['r_edge'], minibatch['r_hood_indices']
        node_l, edge_l, hood_l = minibatch['l_vertex'], minibatch['l_edge'], minibatch['l_hood_indices']
        pair = minibatch['label']
        y = pair[:, 2]
        y[y < 0] = 0
        test_r = model([node_r, edge_r, hood_r, node_l, edge_l, hood_l, pair], training=False).numpy()
        pred_label += test_r.tolist()
        label += y.tolist()
        count += 1

    fpr, tpr, th = metrics.roc_curve(label, pred_label)
    print('test AUC :', metrics.auc(fpr, tpr))

