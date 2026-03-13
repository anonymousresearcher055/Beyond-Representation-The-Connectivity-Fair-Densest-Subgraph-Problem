import tensorflow.compat.v2 as tf
    




class DenseSubgraph(tf.keras.layers.Layer):

    def __init__(self, dropout_rate=0.0, eps=1e-9, tau=1.0, lambda_empty=0.0):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.eps = eps
        self.tau = tau
        self.lambda_empty = lambda_empty

    def build(self, input_shape):
        self.scorer = tf.keras.Sequential([
            tf.keras.layers.Dense(2, kernel_initializer="orthogonal", bias_initializer="zeros"),
            tf.keras.layers.Dropout(self.dropout_rate),
        ])
        super().build(input_shape)

    def call(self, inputs, training=False):
        features, adjacency = inputs  

        logits = self.scorer(features, training=training)        
        probs  = tf.nn.softmax(logits / self.tau, axis=1)        
        p = probs[:, 0:1]                                        


        Ap  = tf.sparse.sparse_dense_matmul(adjacency, p)        
        num = tf.matmul(p, Ap, transpose_a=True)                 
        den = tf.reduce_sum(tf.square(p)) + self.eps
        rayleigh = tf.squeeze(num) / den
        self.add_loss(-rayleigh)


        if self.lambda_empty and self.lambda_empty > 0.0:
            self.add_loss(self.lambda_empty * (-tf.math.log(tf.reduce_sum(p) + self.eps)))

        return probs

      
class DenseSubgraphMaxMin(tf.keras.layers.Layer):

    def __init__(self, dropout_rate=0.0, eps=1e-9,
                 tau=0.5,               
                 tau_min=1, lambda_empty=0.0):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.eps = eps
        self.tau = tau
        self.tau_min = tau_min          
        self.lambda_empty = lambda_empty

    def build(self, input_shape):
        self.scorer = tf.keras.models.Sequential([
            tf.keras.layers.Dense(2, kernel_initializer="orthogonal", bias_initializer="zeros"),
            tf.keras.layers.Dropout(self.dropout_rate),
        ])
        super().build(input_shape)

    def call(self, inputs, training=False):
        emb = inputs[0]
        group_adjs = inputs[1:]  

        logits = self.scorer(emb, training=training)
        probs = tf.nn.softmax(logits / self.tau, axis=1)   
        p = probs[:, 0:1]                                  

        den = tf.reduce_sum(tf.square(p)) + self.eps

        rays = []
        for A_g in group_adjs:
            Ap = tf.sparse.sparse_dense_matmul(A_g, p)
            num = tf.matmul(p, Ap, transpose_a=True)
            rays.append(tf.squeeze(num) / den)
        rays = tf.stack(rays, axis=0) 

        
        tau_min = tf.convert_to_tensor(self.tau_min, dtype=rays.dtype)
        softmin_ray = -tau_min * tf.reduce_logsumexp(-rays / tau_min)
        self.add_loss(-softmin_ray)

        
        
                

        if self.lambda_empty and self.lambda_empty > 0.0:
            self.add_loss(self.lambda_empty / (tf.reduce_sum(p) + self.eps))

        return probs


