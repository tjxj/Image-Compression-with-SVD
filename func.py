import numpy as np

def rebuild_img(u, sigma, v, percent): #p表示奇异值的百分比
    m = len(u)
    n = len(v)
    a = np.zeros((m, n))
 
    #根据指定的清晰度提取奇异值
    #（清晰度越高，压缩比越低，提取的奇异值的个数也就越多，图片也就越不会失真）
    count = (int)(sum(sigma))
    curSum = 0
    k = 0
    while curSum <= count * percent:
        uk = u[:, k].reshape(m, 1)
        vk = v[k].reshape(1, n)
        a += sigma[k] * np.dot(uk, vk)
        curSum += sigma[k]
        k += 1
 
    a[a < 0] = 0
    a[a > 255] = 255
    #按照最近距离取整数，并设置参数类型为uint8
    return np.rint(a).astype("uint8")