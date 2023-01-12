import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from slack_emergency_notifiy import send_slack_msg
import csv

from slack_emergency_notifiy import send_slack_msg

#./model.ckpt.meta
saver = tf.train.import_meta_graph('C:\\csi-motion-main\\csi-motion-main\\model.ckpt.meta')
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("Placeholder:0")
pred = graph.get_tensor_by_name("add:0")
plt.ion()
plt.figure(0)
b = np.arange(0, 15000)
act = ["Walk", "Stand", "Empty", "Sit down", "Stand up"]


with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    ll = np.loadtxt('C:\\csi-motion-main\\csi-motion-main\\aTl101.csv', dtype='float', comments='#', delimiter=',', encoding = 'bytes', max_rows = None )
    ll = np.expand_dims(ll, axis=0)
    saver.restore(sess, 'C:\\csi-motion-main\\csi-motion-main\\model.ckpt')
    n = pred.eval(feed_dict={x: ll})
    n2 = tf.argmax(n, 1)
    result = n2.eval()


    pred_act_label = act[int(result)]
    print(pred_act_label)
    send_slack_msg(pred_act_label)  # 슬랙으로 행동 알림
