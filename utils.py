import numpy as np

class CubicSplineInterpolator:
    
    def __init__(self,xGrid,fGrid):
        
        self.xGrid = xGrid 
        self.coeffs = self.__computeCoefficients(xGrid,fGrid)
        
    def __computeCoefficients(self,xGrid,fGrid):
        N = xGrid.shape[0]
        coeffs = np.zeros((N+1, 4))
        h = np.r_[0, xGrid[1:] - xGrid[:-1], 0]
        l = np.r_[0, (fGrid[1:] - fGrid[:-1]) / (h[1:-1] + 1e-8), 0]
        
        delta = np.zeros((N+1))
        lam = np.zeros((N+1))
        delta[1] = -h[2]/(2*(h[1]+h[2]) + 1e-8)
        lam[1] = 1.5*(l[2]-l[1])/(h[1]+h[2] + 1e-8)
        
        for i in range(2, N):
            delta[i] = -h[i+1] / (2*(h[i] + h[i+1]) + h[i]*delta[i-1] + 1e-8)
            lam[i] = (3*(l[i+1] - l[i]) - h[i]*lam[i-1])/(2*(h[i] + h[i+1]) + h[i]*delta[i-1] + 1e-8)
        
        a = np.r_[fGrid, 0]
        b = np.zeros(N+1)
        c = np.zeros(N+1)
        d = np.zeros(N+1)
        
        coeffs[:,0] = a
        
        for i in range(N, 1, -1):
            c[i-1] = c[i]*delta[i-1] + lam[i-1]
            
        for i in range(1, N+1):
            b[i] = l[i] + (2*c[i]*h[i] + h[i]*c[i-1])/3
            d[i] = (c[i] - c[i-1])/(3*h[i] + 1e-8)
        
        coeffs[:,1] = b
        coeffs[:,2] = c
        coeffs[:,3] = d
        return coeffs
        
    def Compute(self,x):
        res = []
        for x_val in x:
            for i in range(1, len(self.xGrid)):
                if (x_val >= self.xGrid[i-1]) and (x_val <= self.xGrid[i]):
                    diff = x_val - self.xGrid[i]
                    res.append(self.coeffs[i,0] +
                               self.coeffs[i,1]*(diff) +
                               self.coeffs[i,2]*(diff**2) +
                               self.coeffs[i,3]*(diff**3))
                    break
        if len(res) == 1:
            return res[0]
        return np.array(res)
        
    def Integrate(self, a, b):
        ints = 0.0
        i_start = -1
        for i in range(1, len(self.xGrid)):
            if (a >= self.xGrid[i-1]) and (a <= self.xGrid[i]):
                i_start = i
                break
                
        for i in range(i_start, len(self.xGrid) - 1):
            if i == i_start:
                h = self.xGrid[i+1] - self.xGrid[i]
                delta = a - self.xGrid[i]
                ints += self.coeffs[i,3] / 4 * (h**4 - delta**4) + self.coeffs[i,2] / 3 * (h**3 - delta**3) + self.coeffs[i,1] / 2 * (h**2 - delta**2) + self.coeffs[i,0] * (h - delta)
            else:
                ints += self.__integral(i, self.xGrid[i], self.xGrid[i+1])

        return ints
        
    
    def __integral(self, i, a, b):
        delta = b - a
        return self.coeffs[i,3] / 4 * (delta**4) + self.coeffs[i,2] / 3 * (delta**3) + self.coeffs[i,1] / 2 * (delta**2) + self.coeffs[i,0] * delta
    
    def Derivative(self,x):
        res = []
        for x_val in x:
            for i in range(1, len(self.xGrid)):
                if (x_val >= self.xGrid[i-1]) and (x_val <= self.xGrid[i]):
                    diff = x_val - self.xGrid[i]
                    res.append(self.coeffs[i,1] +
                               2*self.coeffs[i,2]*(diff) +
                               3*self.coeffs[i,3]*(diff**2))
                    break
                    
        if len(res) == 1:
            return res[0]
        return np.array(res)

    
def EulerSolver(x_dot, y_dot, x_0, y_0, T, t_0, h, beta=0, y_bound=False):
    
    xs = [x_0]
    ys = [y_0]
    x = x_0
    y = y_0
    ts = np.arange(t_0, T + 1e-9, h)
    
    for i, t in enumerate(ts[:-1]):
        x_vec = x_dot(x, y, t, beta)
        y_vec = y_dot(x, y, t, beta)
        
        assert x_vec == x_vec
        assert y_vec == y_vec
        
        x = x + x_vec * h
        y = y + y_vec * h
        
        if y_bound:
            if y <= 0:
                y = 0.0
            elif y >= 1:
                y = 1.0
        
        xs.append(x)
        ys.append(y)
    
    xs = np.array(xs)
    ys = np.array(ys)
    return CubicSplineInterpolator(ts, xs), CubicSplineInterpolator(ts, ys)
