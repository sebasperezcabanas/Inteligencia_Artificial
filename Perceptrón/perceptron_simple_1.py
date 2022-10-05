import math

def mcCulloch_Pitts(x1, w1, x2, w2, x3, w3, x4, w4, b):
    
    v = x1*w1+x2*w2+x3*w3+x4*w4+b
    
    if (v >= 0):
        return 1
    else:
        return 0

def sigm(x1, w1, x2, w2, x3, w3, x4, w4, b, a):
    
    v = x1*w1+x2*w2+x3*w3+x4*w4+b
    
    return ("%.3f"%(1 / (1+(math.exp(-a*v)))))

def tan_h(x1, w1, x2, w2, x3, w3, x4, w4, b):
    
    v = x1*w1+x2*w2+x3*w3+x4*w4+b
    
    return ("%.3f"%(math.tanh(v/2)))

y = mcCulloch_Pitts(10,0.8,-20,0.2,4,-1.0,-2,-0.9,1)
print(y)

y = sigm(10,0.8,-20,0.2,4,-1.0,-2,-0.9,1,2)
print(y)

y = tan_h(10,0.8,-20,0.2,4,-1.0,-2,-0.9,1)
print(y)
