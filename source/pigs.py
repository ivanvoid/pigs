import numpy as np

class Pigs:
    def __init__(self, stop=1e-5, beta=1, kernel=None, verbose=False):
        self.stop_condition = stop
        self.verbose = verbose
        
        self.beta = beta
        self.kernel = kernel
        self.rec_iteration = 0
        
    
#     def segment(self, image):
        
#         # 01.
#         flat_image = self.flatter(image)
#         # 02.
#         gram = self.compute_gram(flat_image, self.kernel)
        
#         # 06.
#         self._recursion(flat_image, gram, 1)
        
        
    def _recursion(self, img_vector, gram, IR):
        print('Iteration:', self.rec_iteration, '\tIR:',IR)
        # Stop condition 
        if IR < self.stop_condition:
            return 
        
        # 03.
        degree, laplace = self.compute_LD(gram)
        # 04.
        x, root_node = self.linear_solver(degree, laplace)
        # 05.
        threshold, ir = self.compute_IR_threshold(x, gram, degree)
        mask = self.generate_mask(img_vector, x, threshold, root_node)
        
        # 051.
        class_1 = img_vector[mask]
        class_2 = img_vector[np.logical_not(mask)]
        
        gram_1 = None
        gram_2 = None
        
#         submask_1 = self._recursion(class_1, gram_1, ir[0])
#         submask_2 = self._recursion(class_2, gram_2, ir[1])
        
#         mask[mask] = submask_1
#         mask[!mask] = submask_2

        self.rec_iteration += 1
        return mask
    
    def generate_mask(self, img_vector, x, threshold, root_node):
        mask = np.zeros(img_vector.shape[0], dtype=bool)
        mask[:root_node] = x[:root_node] > threshold
        if root_node  == 0:
            mask[root_node] = x[root_node+1] > threshold
        elif root_node  == img_vector.size:
            mask[root_node] = x[root_node+1] > threshold
        else:
            mask[root_node] = x[root_node-1] > threshold
            
        mask[root_node+1:] = x[root_node:] > threshold
        return mask
        
        
#     def flatter(self, image):
#         if type(image)  == np.array:
#             return image.flatten()
#         else:
#             if self.verbose: print('Image shape:', np.array(image).shape)
#             return np.array(image).astype(np.float)[:,:,0].flatten() / 255
        
        
    def compute_gram(self, flat_image, kernel=None, beta=None):
        # Original weights v1
        if beta == None: beta = self.beta
        
        N = flat_image.size
        gram = np.zeros((N,N), dtype=np.float)
        for i in range(N):
            for j in range(N):
                if i < j:
                    if kernel:
                        gram = kernel(flat_image[i], flat_image[j])
                    else:
                        gram[i,j] = self._kernel(
                            flat_image[i], flat_image[j], i, j, 
                            len(flat_image), beta=beta)
        gram += gram.T
        for i in range(N):
            gram[i,i] = 1
        return gram
    
    def _compute_gram(self, flat_image, kernel=None):
        # Low densety graph weights v2
        N = flat_image.size
        gram = np.zeros((N,N), dtype=np.float)
        
        for i in range(N):
            # for even
            if i%2 == 0:
                for j in range(N//2):
                    ii = i
                    jj = j*2+1 # odd
                    if ii < jj:    
                        gram[ii,jj] = self._kernel(
                            flat_image[ii], flat_image[jj], ii, jj, 
                            len(flat_image), beta=self.beta)
            # for odd
            else:
                for j in range(N//2):
                    ii = i
                    jj = j*2 # even
                    if ii < jj:        
                        gram[ii,jj] = self._kernel(
                            flat_image[ii], flat_image[jj], ii, jj, 
                            len(flat_image), beta=self.beta)
                    
        gram += gram.T
        return gram
    
    def _compute_gram(self, flat_image, diag_size=None, kernel=None):
        # Low densety diagonal graph weights v3
        N = flat_image.size
        gram = np.zeros((N,N), dtype=np.float)
        # distance from main diaganal to computational boundory
        if diag_size == None: 
            diag_size = N // 2
        print("Diagonal size:", diag_size)
        
        for i in range(N):
            # where to compute
            M = i + diag_size
            if N - M < 0:
                M = M + (N-M)
            # for even
            if i%2 == 0:
                for j in range(i, M, 3):
                    ii = i 
                    jj = j
                    if ii < jj:
                        gram[ii,jj] = self._kernel(
                            flat_image[ii], flat_image[jj], 
                            ii, jj, N, beta=self.beta)
            # odd
            else:
                for j in range(i, M, 2):
                    ii = i; 
                    jj = j # even
                    if ii < jj:
                        gram[ii,jj] = self._kernel(
                            flat_image[ii], flat_image[jj], ii, jj, 
                            len(flat_image), beta=self.beta)
                
        gram += gram.T
        return gram
    
    
    def _kernel(self, x, y, i, j, N, beta=3):
        val = (x - y)**2
        # V1
        pos = ((i - j)/(N**2))**2
        weight = val + pos

        # V2
#         pos = 1 - ((i - j)/(N**2))**2
#         weight = val * pos
        
        weight = np.exp(-beta * weight)
        return weight     
    
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
        if self.verbose: print('Root node:',root_node)
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
        
        return x, root_node
            
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
        iso_ratio = perimeter / (volume+1e-9)
        return iso_ratio
    