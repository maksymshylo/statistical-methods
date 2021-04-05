import argparse
import numpy as np
import matplotlib.pyplot as plt
import random
from gibbs_sampler_grid import (create_image, 
     apply_noise,
     get_output_image,
     get_accuracy,
     generate_lines,
    )


def find_cols(noised_image, p, pc_r, pc_c):
    
    """
    compute precise columns for relatively small images  

    p - noise level
    pc_r, pc_c - probability of row or column to be black
    """

    height, width = noised_image.shape
    # labels of column 0 or 1
    labels = np.arange(2)
    # column probability
    cols_probabs = np.zeros((2,width))
    # temporary array of probabilities to get image with given columns and rows
    conditional_probab_array = np.zeros((2,height), dtype=np.float64)
    cond_probab = np.zeros((2))
    for c in labels:
        for j in range(width):
            for r_j in labels:
                for i in range(height):
                    conditional_probab_array[r_j,i] = (noised_image[i,j]^(c or r_j))*p + (noised_image[i,j]^(1 - c or r_j))*(1-p)
                cond_probab[r_j] = np.prod(conditional_probab_array[r_j,:])*pc_r[r_j]
            # sum of all possible values of r_j
            cols_probabs[c,j] = np.sum(cond_probab)
    # precise position of black columns
    cols = np.zeros((width),dtype=int)
    # get columns  by generating from distribution
    for j in range(width):
        cols[j] = np.random.choice(labels,p=cols_probabs[:,j]/np.sum(cols_probabs[:,j]))
    return cols


def precise_solution(noised_image, p, pc):

    """
    calculates rows and cols positions

    p - noise level
    pc_r, pc_c - probability of row or column to be black

    """

    pc_r = pc**np.arange(2)*(1-pc)**(1-np.arange(2))
    pc_c = pc**np.arange(2)*(1-pc)**(1-np.arange(2))
    cols = find_cols(noised_image, p, pc_r, pc_c)
    # generating rows by fixing columns
    rows = generate_lines(image=noised_image,
                          fixed_generated=cols,
                          p=p,
                          apriori_prob=pc_c,
                          object_="rows")

    return cols, rows



def main():
    # parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("height", type=int, help="height of input_image")
    parser.add_argument("width", type=int, help="width of input_image")
    parser.add_argument("n_generated_lines", type=int, help="number of horizontal and vertical lines")
    parser.add_argument("noise_level", type=float, help="noise level of bernoulli distribution")
    parser.add_argument("column_probab", type=float, help="probability of column to be black")

    args = parser.parse_args()

    image, true_cols, true_rows = create_image(args.height, args.width, args.n_generated_lines)
    noised_image = apply_noise(image, args.noise_level)
    cols, rows = precise_solution(noised_image, args.noise_level, args.column_probab)
    cols_acc, rows_acc = get_accuracy(true_cols, true_rows, cols, rows)
    output_image = get_output_image(cols, rows)
    print("column accuracy", cols_acc)
    print("row accuracy",    rows_acc)

    plt.imsave("input_image_precise.png",image, cmap = 'binary')
    plt.imsave("noised_image_precise.png",noised_image, cmap = 'binary')
    plt.imsave("output_image_precise.png",output_image, cmap = 'binary')



if __name__ == "__main__":
    main()
