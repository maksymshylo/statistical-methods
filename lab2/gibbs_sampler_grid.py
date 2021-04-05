import argparse
import numpy as np
import matplotlib.pyplot as plt
import random
from numba import njit

def create_image(height, width, n_generated_lines):

    """
    height, width - parameters of input image
    n_generated_lines - number of vertical and horizontal lines to be black in input image(chooses randomly)
    """
    image = np.zeros((height,width),dtype=int)


    # true_rows, true_cols - true values of black rows and columns accordingly
    true_rows= np.zeros((height,),dtype=int)
    true_cols= np.zeros((width,),dtype=int)

    rows_indices = random.sample(range(0,height),n_generated_lines)
    cols_indices = random.sample(range(0,width),n_generated_lines)

    true_rows[rows_indices] = 1
    true_cols[cols_indices] = 1

    image[:,cols_indices]   = 1
    image[rows_indices,:]   = 1

    return image, true_cols, true_rows

def apply_noise(image, p):

    """
    image - input image
    p - noise level of bernoulli distribution
    """
    ksi = np.random.binomial(size=image.size, n=1, p=p).reshape(image.shape)
    noised_image = ksi^image

    return noised_image

@njit
def random_choice(distribution):

    # generate random number from a given distribution
    r = random.uniform(0, 1)
    s = 0
    for item, prob in enumerate(distribution):
        s += prob
        if s >= r:
            return item
    return item  

@njit
def generate_lines(image, fixed_generated, p, apriori_prob, object_ ):
    
    """
    generate lines (horizontal or vertical) with fixed opposite lines

    image - input image
    fixed_generated - fixed line (what line - defines parameter object_ ("rows" or "cols"))
    p - noise level of bernoulli distribution
    apriori_prob - apriori probability for set of rows (columns) to be black
    """
    height, width = image.shape

    labels = np.arange(2)
    # oppoiste shapes for cols and rows cases
    if object_ == "cols":
        dim1,dim2 = width, height
    elif object_ == "rows":
        dim1,dim2 = height, width
        image = image.T
    generated_lines = np.zeros(dim1,dtype=np.int64)   
    
    for j in range(dim1): 
        # conditional probability for image with fixed rows and columns
        conditional_probab_array = np.zeros((2,dim2), dtype=np.float64)
        for i in range(dim2):
            label = np.logical_or(labels, fixed_generated[i])
            conditional_probab_array[:,i] = (image[i,j]^(label))*p + (image[i,j]^(1 - label))*(1-p)
        conditional_probab = np.array([np.prod(conditional_probab_array[0,:]), 
                                       np.prod(conditional_probab_array[1,:])])
        # probability of row (column) to be black (with opposite fixed)
        row_or_col_prob = conditional_probab*apriori_prob
        generated_lines[j] = random_choice(row_or_col_prob/np.sum(row_or_col_prob))
    # returns generated lines
    return generated_lines


def gibbs_sampler(noised_image, p, pc, n_iter):
    
    """
    generate lines with fixed cols or rows
    
    p - noise level of bernoulli distribution

    pc - appriori probability of column to be black

    n_iter - number of iterations for Gibbs Sampler
    """
    height = noised_image.shape[0]
    # fix rows
    generated_rows = np.random.randint(2,size = height)
    # calculate apriori probability for set of rows (columns) to be black
    apriori_prob = pc**np.arange(2)*(1-pc)**(1-np.arange(2))
    # run Gibbs Sampler
    for _ in range(n_iter):
        # generate cols with fixed rows
        generated_cols = generate_lines(noised_image, fixed_generated = generated_rows, p=p, apriori_prob = apriori_prob, object_ = "cols")
        # generate rows with fixed cols
        generated_rows = generate_lines(noised_image, fixed_generated = generated_cols, p=p, apriori_prob = apriori_prob, object_ = "rows")

    return generated_cols, generated_rows



def get_output_image(generated_cols, generated_rows):
    
    """
    create image from generated_cols and generated_rows
    """
    height,width = len(generated_rows), len(generated_cols)

    output_img = np.zeros((height,width),dtype=int)
    out_cols_ind = np.arange(0,width)[generated_cols!=0]
    out_rows_ind = np.arange(0,height)[generated_rows!=0]
    output_img[:,out_cols_ind] = 1
    output_img[out_rows_ind,:] = 1

    return output_img

def get_accuracy(true_cols, true_rows, generated_cols, generated_rows):
    """
    calculates percentage of matched lines
    """
    cols_acc = ((generated_cols == true_cols).sum()/len(true_cols))*100
    rows_acc =((generated_rows == true_rows).sum()/len(true_rows))*100
    return cols_acc, rows_acc

def main():
    # parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("height", type=int, help="height of input_image")
    parser.add_argument("width", type=int, help="width of input_image")
    parser.add_argument("n_generated_lines", type=int, help="number of horizontal and vertical lines")
    parser.add_argument("noise_level", type=float, help="noise level of bernoulli distribution")
    parser.add_argument("column_probab", type=float, help="probability of column to be black")
    parser.add_argument("n_iter", type=int, help="number of iterations for Gibbs Sampler")

    args = parser.parse_args()

    image, true_cols, true_rows = create_image(args.height, args.width, args.n_generated_lines)
    noised_image = apply_noise(image, args.noise_level)
    generated_cols, generated_rows = gibbs_sampler(noised_image, args.noise_level, args.column_probab, args.n_iter)
    cols_acc, rows_acc = get_accuracy(true_cols, true_rows, generated_cols, generated_rows)
    output_image = get_output_image(generated_cols, generated_rows)
    print("column accuracy", cols_acc)
    print("row accuracy",    rows_acc)

    plt.imsave("input_image_gibbs.png",image, cmap = 'binary')
    plt.imsave("noised_image_gibbs.png",noised_image, cmap = 'binary')
    plt.imsave("output_image_gibbs.png",output_image, cmap = 'binary')



if __name__ == "__main__":
    main()
