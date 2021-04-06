import sys, os, argparse
import json, time, string
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from decimal import *

EPS = 10**-320

def import_images(folder_path,alphabet_list):
    """
    reads images from folder_path and creates set of images
    """
    reference_images = {}
    for i in alphabet_list[:-1] + ['space']:
        img = np.array(Image.open(folder_path + f'/{i}.png')).astype('int')
        reference_images[i] = img
    reference_images[' '] = reference_images.pop('space')
    return reference_images

def string_to_image(string,reference_images,noise_level):
    """
    convertes string to image and adds noise
    """
    # create string as array
    image = reference_images[string[0]]
    for i in string[1:]:
        image = np.hstack([image,reference_images[i]])
    n,m = image.shape
    # generate binomial noise
    ksi = np.random.binomial(size=n*m, n=1, p=noise_level).reshape(image.shape)
    output_image = ksi^image
    return output_image

def calculate_tail(input_image,p,letters,p_k):
    """
    calculates sum of probabilities for the rest of the string
    """
    n = input_image.shape[1]
    # array of sums of probabilities
    sum_of_probab = np.full((n+1),0,dtype=Decimal)
    # last element always equals 1
    sum_of_probab[-1] = 1
    for s in range(n-1,-1,-1):
        inner_sum = np.full((len(letters)),Decimal(0).normalize(),dtype=Decimal)
        for i,letter in enumerate(letters):
            # the end of possible letter
            cut_till = s + letter.shape[1]
            # calculate conditional probability
            if cut_till <= n:
                x = input_image[:,s:cut_till]
                if p == 0:
                    cond_probab = np.sum( (x^letter)*np.log(p + EPS))  + p_k
                elif p == 1:
                    cond_probab = np.sum((1^x^letter)*np.log(EPS))  + p_k
                else:
                    cond_probab = np.sum( (x^letter)*np.log(p) + (1^x^letter)*np.log(1-p)   )  + p_k
                # add to inner sum
                inner_sum[i] = Decimal(cond_probab).exp()*Decimal(sum_of_probab[cut_till])
        sum_of_probab[s] = np.sum(inner_sum)
    return sum_of_probab

def letter_probab(input_image,p,letters,sum_of_probab,p_k):
    """
    calculates probability of the letter p(x)
    """
    n = input_image.shape[1]
    letter_probab = np.zeros(len(letters),dtype=Decimal)
    for i,letter in enumerate(letters):
        cut_till = letter.shape[1]
        if cut_till <= n:
            x = input_image[:,:cut_till]
            if p == 0:
                cond_probab = np.sum( (x^letter)*np.log(p + EPS))  + p_k
            elif p == 1:
                cond_probab = np.sum((1^x^letter)*np.log(EPS))  + p_k
            else:
                cond_probab = np.sum( (x^letter)*np.log(p) + (1^x^letter)*np.log(1-p))  + p_k
            letter_probab[i] = Decimal(cond_probab).exp()*Decimal(sum_of_probab[cut_till])
    return letter_probab

def generete_string(input_image,p,letters,p_k,alphabet_list):
    """
    generate string from image
    """
    output_string = []
    width_prev_letter = 0
    while input_image[:,width_prev_letter:].size != 0:
        # cut input_image based on previous letter
        input_image = input_image[:,width_prev_letter:]
        # calculate probabilities for the first letter
        sum_of_probab = calculate_tail(input_image,p,letters,p_k)
        pk1 = letter_probab(input_image,p,letters,sum_of_probab,p_k)
        # generate letter from probability
        generated_letter = np.random.choice(len(pk1), 1, p=list(pk1/pk1.sum()))[0]
        width_prev_letter = letters[generated_letter].shape[1]
        # get generated letter
        output_string.append(alphabet_list[generated_letter])
    return ''.join(output_string)


def main():
    # parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("input_string", type=str, help="input string")
    parser.add_argument("noise_level", type=float, help="noise level of bernoulli distribution")
    args = parser.parse_args()
    alphabet_list = list(string.ascii_lowercase + ' ')
    reference_images = import_images("alphabet",alphabet_list)
    
    input_image =  string_to_image(args.input_string,reference_images,noise_level=0)
    noised_image = string_to_image(args.input_string,reference_images,args.noise_level)
    letters = list(reference_images.values())
    p_k = np.log(1/len(alphabet_list))
    output_string = generete_string(noised_image,args.noise_level,letters,p_k,alphabet_list)
    output_image = string_to_image(output_string,reference_images,noise_level=0)
    print(output_string)

    plt.imsave("input_image.png", input_image, cmap ="binary")
    plt.imsave("noised_image.png", noised_image, cmap ="binary")
    plt.imsave("output_image.png", output_image, cmap ="binary")



if __name__ == "__main__":
    main()
