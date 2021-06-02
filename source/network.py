import tensorflow as tf
import tensorflow.keras.backend as K

def make_gcn_layer(input_dim, output_dim):
    nodes = tf.keras.Input(shape=(input_dim))#, name='node')
    edges = tf.keras.Input(shape=(20, 2))#, name='edge')
    hoods = tf.keras.Input(shape=(20, 1))#, name='hood')
    
    # flatten 
    edges_f = tf.keras.layers.Flatten()(edges)
    
    wc = tf.keras.layers.Dense(output_dim)(nodes)
    wn = tf.keras.layers.Dense(output_dim, use_bias=False)(nodes)
    we = tf.keras.layers.Dense(output_dim, use_bias=False)(edges_f)
    
    x = wc + (K.mean(wn * we))#K.sum(K.abs(hoods)))#K.sum(K.abs(hoods)))#K.sum(tf.maximum(hoods, tf.ones_like(hoods))))#K.sum(K.abs(hoods)))
    outputs = tf.keras.layers.Activation('linear')(x)
    return tf.keras.Model([nodes, edges, hoods], outputs)#, name='gcn')

def make_pipgcn(input_dim, hidden_dim, output_dim):
    # define data input
    nodes_r = tf.keras.Input(shape=(input_dim,), name='node_r')
    edges_r = tf.keras.Input(shape=(20, 2), name='edge_r')
    hoods_r = tf.keras.Input(shape=(20, 1), name='hood_r')
    nodes_l = tf.keras.Input(shape=(input_dim,), name='node_l')
    edges_l = tf.keras.Input(shape=(20, 2), name='edge_l')
    hoods_l = tf.keras.Input(shape=(20, 1), name='hood_l')
    pair = tf.keras.Input(shape=(3,), name='pair', dtype='int32')
    
    # set filter 
    filters = [48, 36]#[256, 256, 512, 512]
    inputs =  [input_dim] + filters
    gconv = []
    for i, f in enumerate(filters):
        gconv.append(make_gcn_layer(inputs[i], f))
        
    rr = nodes_r
    lr = nodes_l
    for i, f in enumerate(filters):
        rr = gconv[i]([rr, edges_r, hoods_r]) # 326
        lr = gconv[i]([lr, edges_l, hoods_l]) # 326
    

    # select top 20 strong connection
    rr = tf.gather(rr, pair[:, 1], name='gather_r', axis=0)
    lr = tf.gather(lr, pair[:, 0], name='gather_l', axis=0)
    
    # merge 
    x = tf.concat([rr, lr], axis=-1)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(output_dim, activation='sigmoid', name='sof')(x)

    return tf.keras.Model([nodes_r, edges_r, hoods_r, nodes_l, edges_l, hoods_l, pair], outputs, name='clf')   
