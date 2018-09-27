import numpy as np
import brica


class SC(object):
    """ 
    SC (superior colliculus) module.
    SC outputs action for saccade eye movement.
    """
    def __init__(self):
        self.timing = brica.Timing(6, 1, 0)

        self.last_fef_data = None
        self.last_sc_data = None
        self.baseline = None

    def __call__(self, inputs):
        if 'from_fef' not in inputs:
            raise Exception('SC did not recieve from FEF')
        if 'from_bg' not in inputs:
            raise Exception('SC did not recieve from BG')

        # Likelihoods and eye movment params from accumulators in FEF module.
        fef_data = inputs['from_fef']
        # Likelihood thresolds from BG module.
        bg_data = inputs['from_bg']

        action = self._decide_action(fef_data, bg_data)
        
        # Store FEF data for debug visualizer
        self.last_fef_data = fef_data
        
        return dict(to_environment=action)

    def _decide_action(self, fef_data, bg_data):
        sum_ex = 0.0
        sum_ey = 0.0

        def mixture_gauss(params):
            params = params * 8
            print(params)
            mu1 = params[[0,1]]
            mu2 = params[[4,5]]
            det1 = (params[2]+0.1)*(params[3]+0.1)
            det2 = (params[6]+0.1)*(params[7]+0.1)
            inv1 = np.array([[1/(params[2]+0.1), 0], [0, 1/(params[3]+0.1)]])
            inv2 = np.array([[1/(params[6]+0.1), 0], [0, 1/(params[7]+0.1)]])
            lam1 = params[8]
            lam2 = params[9]

            def f(x, y):
                x_c1 = np.array([x, y]) - mu1
                exp1 = np.exp(- np.dot(np.dot(x_c1,inv1),x_c1[np.newaxis, :].T) / 2.0) 
                x_c2 = np.array([x, y]) - mu2
                exp2 = np.exp(- np.dot(np.dot(x_c2,inv2),x_c2[np.newaxis, :].T) / 2.0) 
                return lam1*exp1/(2*np.pi*np.sqrt(det1)) + lam2*exp2/(2*np.pi*np.sqrt(det2))

            x = y = np.arange(0,8)
            X, Y = np.meshgrid(x, y)#y axis same, x axis same
            Z = np.vectorize(f)(X,Y)
            return Z.reshape(-1)
        '''
        #assert(len(fef_data) == len(bg_data))

        count = 0

        # Calculate average eye ex, ey with has likelihoods over
        # the thresholds from BG.
        diff = fef_data[0:, 0] - bg_data
        max_idx = np.argmax(diff)
        action = fef_data[max_idx, 1:]
        '''
        #print(bg_data)
        self.baseline = mixture_gauss(bg_data)
        print(self.baseline)
        print(fef_data)
        diff = fef_data[:,0]+(self.baseline/8.0)
        self.last_sc_data = diff
        #print(diff)
        max_idx = np.argmax(diff)
        action = fef_data[max_idx, 1:]
        #print('action', action)
        return action

        '''
        for i,data in enumerate(fef_data):
            likelihood = data[0]
            ex = data[1]
            ey = data[2]
            likelihood_threshold = bg_data[i]
            
            if likelihood > likelihood_threshold:
                sum_ex += ex
                sum_ey += ey
                count += 1       
        # Action values should be within range [-1.0~1.0]
        if count != 0:
            action = [sum_ex / count, sum_ey / count]
        else:
            action = [0.0, 0.0]
        '''
        #return np.array(action, dtype=np.float32)
