from xml.dom.pulldom import default_bufsize
import cv2
import numpy as np
from PIL import Image
import gradio as gr
from func import rebuild_img


def inference(img,k):
    input_img = cv2.imread(img, cv2.IMREAD_COLOR)    
    #k=gr.inputs.Slider(0, 1, 0.1)
    u, sigma, v = np.linalg.svd(input_img[:, :, 0])
    R = rebuild_img(u, sigma, v, k)
    u, sigma, v = np.linalg.svd(input_img[:, :, 1])
    G = rebuild_img(u, sigma, v, k)
    u, sigma, v = np.linalg.svd(input_img[:, :, 2])
    B = rebuild_img(u, sigma, v, k)
    restored_img = np.stack((R, G, B), 2)
    #return Image.fromarray(restored_faces[0][:,:,::-1])
    return Image.fromarray(restored_img[:, :, ::-1])
    

title = "用 SVD 压缩图片"

description = "上传需要压缩的图片，选择压缩比，点击Submit，稍等片刻，右侧Output将照片另存为即可。"

article = "<p style='text-align: center'><a href='https://mp.weixin.qq.com/s?__biz=MzA4MjYwMTc5Nw==&amp;mid=2648965531&amp;idx=1&amp;sn=e63b762c12182f74f077df0aa0e7bb53&amp;chksm=879393b1b0e41aa720d7cf54b3c5eac4d26cb4414522d7d5bec67eef871bdee06b4562e33bc4&token=242929914&lang=zh_CN#rd' target='_blank'>SVD 简介</a> | <a href='https://github.com/tjxj/100-Days-Of-ML-Code' target='_blank'>100天搞定机器学习</a></p><center><img src='https://visitor-badge.glitch.me/badge?page_id=akhaliq_GFPGAN' alt='visitor badge'></center>"


gr.Interface(
    inference, 
    [
    gr.inputs.Image(type="filepath", label="Input"),gr.inputs.Slider(0, 1, 0.1,default=0.6,label= 'Compression ratio')], 
    gr.outputs.Image(type="pil", label="Output"),
    title=title,
    description=description,
    article=article
    ).launch(enable_queue=True,cache_examples=True,share=True)
    