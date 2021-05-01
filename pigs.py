import numpy as np

class Pigs:
    def __init__(self, stop=1e-5, beta=1, kernel=None, verbose=False):
        self.stop_condition = stop
        self.verbose = verbose
        
        self.beta = beta
        self.kernel = kernel
        self.rec_iteration = 0
        
    
    
    def segment(self, image):
        
        # 01.
        flat_image = self.flatter(image)
        # 02.
        gram = self.compute_gram(flat_image, self.kernel)
        
        # 06.
        self._recursion(flat_image, gram, 1)
        
        
    def _recursion(self, img_vector, gram, IR):
        print('Iteration:', self.rec_iteration, '\tIR:',IR)
        # Stop condition 
        if IR < self.stop_condition:
            return 
        
        # 03.
        degree, laplace = self.compute_LD(gram)
        # 04.
        x = self.linear_solver(degree, laplace)
        # 05.
        threshold, ir = self.compute_IR_threshold(x, gram, degree)
        mask = np.zeros(img_vector.size)
        mask = x > threshold
        
#         # 051.
        class_1 = img_vector[mask]
        class_2 = img_vector[!mask]
        
        gram_1 = None
        gram_2 = None
        
#         submask_1 = self._recursion(class_1, gram_1, ir[0])
#         submask_2 = self._recursion(class_2, gram_2, ir[1])
        
#         mask[mask] = submask_1
#         mask[!mask] = submask_2

        self.rec_iteration += 1
        return mask
        
        
    def flatter(self, image):
        if type(image)  == np.array:
            return image.flatten()
        else:
            if self.verbose: print('Image shape:', np.array(image).shape)
            return np.array(image).astype(np.float)[:,:,0].flatten() / 255
        
        
    def compute_gram(self, flat_image, kernel=None):
        N = flat_image.size
        gram = np.zeros((N,N), dtype=np.float)
        for i in range(N):
            for j in range(N):
                if kernel:
                    gram = kernel(flat_image[i], flat_image[j])
                else:
                    gram[i,j] = self._kernel(flat_image[i], flat_image[j], i, j, beta=self.beta)
        return gram
    
    
    def _kernel(self, x, y, i, j, beta=3):
        return np.exp(-beta * ((x - y)**2 + (i - j)**2))
    
    
    def compute_LD(self, gram):
        degree = np.zeros(gram.shape[0])
        laplace = -1 * gram.copy()
        
        for i in range(gram.shape[0]):
            degree[i] = gram[i].sum() - gram[i,i]
            laplace[i,i] = degree[i]
        
        return degree, laplace
    
    
    def linear_solver(self, degree, laplace):
        # 141.
        root_node = degree.argmax()
        print('Root node:',root_node)
        # 142.
        laplace =  np.concatenate((
            np.concatenate((
                laplace[:root_node,:root_node],
                laplace[:root_node,root_node+1:]),1),
            np.concatenate((
                laplace[root_node+1:,:root_node],
                laplace[root_node+1:,root_node+1:]), 1)))

        degree = np.concatenate((
            degree[:root_node], 
            degree[root_node+1:]))
        
        # 143.
        x = np.linalg.solve(laplace, degree)
        
        return x
            
    def compute_IR_threshold(self, x, gram, degree, steps=20):
        """Greedy threshold search
        Best threshold for lowest Isometric Ratio
        """
        xmin = x.min()
        xmax = x.max()
        step = (xmax - xmin)/steps
        
        lower_th = None
        lower_IR = 1

        # for each potential threshold
        for threshold in np.arange(xmin+step, xmax, step):
            IR = self._compute_isometric_ratio(gram, degree, x, threshold)
            
            if IR < lower_IR:
                lower_IR = IR
                lower_th = threshold
                
        return lower_th, lower_IR 
        
        
    def _compute_isometric_ratio(self, gram, degree, x, threshold):
        """Compute Isometric Ratio of a graph
        
        Isometric Ratio - sum of edge weights divided by
        volume of one of two sets(graphs) seporated by threshold
        """
        perimeter = 0 # dS
        volume = 0

        # A - inside set; B - outside set
        set_A = x >= threshold
        set_B = x < threshold
        
        for a_idx in np.where(set_A)[0]:
            # Compute inside set volume
            volume += degree[a_idx]

            for b_idx in np.where(set_B)[0]:
                # Compute boundaey betwen inside and outside sets
                perimeter += gram[a_idx, b_idx]
        
        # Compute isometric ratio
        iso_ratio = perimeter / volume
        return iso_ratio